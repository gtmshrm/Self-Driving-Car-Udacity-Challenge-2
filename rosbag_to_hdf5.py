#!/usr/bin/python
"""
Convert ROS bag data into hdf5 files with specified batch generation time

Note: Batch generation time of 300 seconds generates batches of 300 images 
(including left, center and right camera frames) with corresponding steering angles
"""
import argparse
import sys
import numpy as np
import rosbag
import datetime
import h5py
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os


def save_batch(dataset, steps, outdir, batch_nbr):
	"""
	Save batch in hdf5 file
	"""
	with h5py.File(outdir+"/batch%d.h5" % (batch_nbr), 'w') as f:
		f.create_dataset('left_camera_image', data=np.array(dataset['left_camera']))
		f.create_dataset('left_camera_steering_angle', data=np.array(dataset['left_camera/steering_angle']))
		f.create_dataset('center_camera_image', data=np.array(dataset['center_camera']))
		f.create_dataset('center_camera_steering_angle', data=np.array(dataset['center_camera/steering_angle']))
		f.create_dataset('right_camera_image', data=np.array(dataset['right_camera']))
		f.create_dataset('right_camera_steering_angle', data=np.array(dataset['right_camera/steering_angle']))
	return batch_nbr + 1
	

# ***** main loop *****
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='ROS bag to hdf5 files converter')
	parser.add_argument('--dataset_path', type=str, default="ros_dataset/dataset.bag", help='Dataset/ROS Bag name')
	parser.add_argument('--skip', type=int, default="0", help='Skip seconds')
	parser.add_argument('--batch_gen_time', type=int, default="-1", help='Time for each batch generation in seconds')
	parser.add_argument('--outdir', type=str, default="hdf5_dataset", help='Output directory')
	args = parser.parse_args()

	if args.batch_gen_time == -1:
		raise Exception("You didn't enter time duration of each batch generation")

	dataset = {'left_camera': [], 
			   'left_camera/steering_angle': [],
			   'center_camera': [], 
			   'center_camera/steering_angle': [],
			   'right_camera': [], 
			   'right_camera/steering_angle': []}

	batch_nbr = 0
	dataset_path = args.dataset_path
	skip = args.skip
	startsec = 0
	batch_gen_time = args.batch_gen_time
	outdir = args.outdir
	bridge = CvBridge()
	steps = 0 
	current_time = 0

	print("reading rosbag ", dataset_path)
	bag = rosbag.Bag(dataset_path, 'r')
	for topic, msg, t in bag.read_messages(topics=['/center_camera/image_color','/right_camera/image_color','/left_camera/image_color','/vehicle/steering_report']):
		print("\nStep: %d" % (steps))
		print("Time: %.2f" % (t.to_sec()))
		current_time = t.to_sec()

		# initialize start time
		if startsec == 0:
			startsec = t.to_sec()
			if skip < 24*60*60:
				skipping = current_time + skip
				print("skipping ", skip, " seconds from ", startsec, " to ", skipping, " ...")
			else:
				skipping = skip
				print("skipping to ", skip, " from ", startsec, " ...")
		
		# within batch generation time slot
		elif current_time > skipping and current_time <= skipping + batch_gen_time and batch_gen_time != -1:
			if topic in ['/vehicle/steering_report']:
				angle_steers = msg.steering_wheel_angle
			try:
				if topic in ['/center_camera/image_color','/right_camera/image_color','/left_camera/image_color']:
					print(topic, msg.header.seq, t-msg.header.stamp, msg.height, msg.width, msg.encoding, t)
					cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
					cv_image = cv2.resize(cv_image, (320, 240), interpolation=cv2.INTER_CUBIC)
					dataset[topic.split('/')[1]].append(cv_image)
					dataset[topic.split('/')[1]+'/steering_angle'].append(angle_steers)
			except Exception, e:
				print("Error saving image.", e)
		
		# batch generation time slot over. make new time slot for new batch
		elif current_time > skipping and current_time > skipping + batch_gen_time and batch_gen_time != -1:
			print("********* NEW BATCH *********")
			batch_nbr = save_batch(dataset, steps, outdir, batch_nbr)
			# empty arrays of prev batch
			dataset = {'left_camera': [], 
					   'left_camera/steering_angle': [],
					   'center_camera': [], 
					   'center_camera/steering_angle': [],
					   'right_camera': [], 
					   'right_camera/steering_angle': []}
			# increase batch generation time to generate another batch
			batch_gen_time += args.batch_gen_time
			# generate first example for new batch
			if topic in ['/vehicle/steering_report']:
				angle_steers = msg.steering_wheel_angle
			try:
				if topic in ['/center_camera/image_color','/right_camera/image_color','/left_camera/image_color']:
					print(topic, msg.header.seq, t-msg.header.stamp, msg.height, msg.width, msg.encoding, t)
					cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
					cv_image = cv2.resize(cv_image, (320, 240), interpolation=cv2.INTER_CUBIC)
					dataset[topic.split('/')[1]].append(cv_image)
					dataset[topic.split('/')[1]+'/steering_angle'].append(angle_steers)
			except Exception, e:
				print("Error saving image.", e)
		
		steps += 1

	# save last batch
	save_batch(dataset, steps, outdir, batch_nbr)