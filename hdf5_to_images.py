"""
Save images from hdf5 files and log image names and corresponding steering angle

"""

import h5py
import argparse
import sys
import numpy as np
import os
import glob
import scipy.misc

# ***** main loop *****
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='hdf5 files to images converter')
	parser.add_argument('--dataset_path', type=str, default="./hdf5_dataset", help='Dataset/ROS Bag name')
	parser.add_argument('--outdir', type=str, default="./driving_dataset", help='Output directory')
	args = parser.parse_args()

	# get list of files
	dfiles = glob.glob(args.dataset_path+"/*.h5")

	# save center images and log the labels into data.txt
	if not os.path.exists(args.outdir+"/center"):
		os.makedirs(args.outdir+"/center")
	print("******* SAVING CENTER CAM IMAGES *******")
	with open(args.outdir+"/center/data.txt", "w") as data_file:
		image_name = 0
		for dfile in dfiles:
			h =  h5py.File(dfile, 'r')
			cam_center = h.get('center_camera_image')
			steering_angle_center = h.get('center_camera_steering_angle')
			for img, steering_angle in zip(cam_center, steering_angle_center):
				print("Saving image %d..." % (image_name))
				# save image
				scipy.misc.imsave(args.outdir+"/center/%d.jpg" % (image_name), img)
				# save image name and steering angle in data.txt
				data_file.write(args.outdir+"/center/{0}.jpg {1}\n".format(image_name, steering_angle))
				image_name += 1
	# free up memory
	del cam_center
	del steering_angle_center

	# save left images and log the labels into data.txt
	if not os.path.exists(args.outdir+"/left"):
		os.makedirs(args.outdir+"/left")
	print("******* SAVING LEFT CAM IMAGES *******")
	with open(args.outdir+"/left/data.txt", "w") as data_file:
		image_name = 0
		for dfile in dfiles:
			h =  h5py.File(dfile, 'r')
			cam_left = h.get('left_camera_image')
			steering_angle_left = h.get('left_camera_steering_angle')
			for img, steering_angle in zip(cam_left, steering_angle_left):
				print("Saving image %d..." % (image_name))
				# save image
				scipy.misc.imsave(args.outdir+"/left/%d.jpg" % (image_name), img)
				# save image name and steering angle in data.txt
				data_file.write(args.outdir+"/left/{0}.jpg {1}\n".format(image_name, steering_angle))
				image_name += 1
	# free up memory
	del cam_left
	del steering_angle_left

	# save right images and log the labels into data.txt
	if not os.path.exists(args.outdir+"/right"):
		os.makedirs(args.outdir+"/right")
	print("******* SAVING RIGHT CAM IMAGES *******")
	with open(args.outdir+"/right/data.txt", "w") as data_file:
		image_name = 0
		for dfile in dfiles:
			h =  h5py.File(dfile, 'r')
			cam_right = h.get('right_camera_image')
			steering_angle_right = h.get('right_camera_steering_angle')
			for img, steering_angle in zip(cam_right, steering_angle_right):
				print("Saving image %d..." % (image_name))
				# save image
				scipy.misc.imsave(args.outdir+"/right/%d.jpg" % (image_name), img)
				# save image name and steering angle in data.txt
				data_file.write(args.outdir+"/right/{0}.jpg {1}\n".format(image_name, steering_angle))
				image_name += 1

