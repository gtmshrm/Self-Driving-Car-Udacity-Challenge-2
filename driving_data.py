from __future__ import with_statement
import scipy.misc
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import augmentation as aug


xs = []
ys = []

# points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0
FDIR = '/Volumes/Untitled/Udacity datasets/challenge 2 training/Ch2_002/images & interpolated data/'
df = pd.read_csv('/Volumes/Untitled/Udacity datasets/challenge 2 training/Ch2_002/images & interpolated data/interpolated.csv')
df = df[df.frame_id == 'center_camera']
xs = [FDIR + s for s in df.filename]
ys = df.angle

# get number of images
num_images = len(xs)

# shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)


def LoadTrainBatch(batch_size):
  global train_batch_pointer
  x_out = []
  y_out = []
  for i in range(0, batch_size):
    # read
    img = cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images])
    # random shifts and rotations
    distorted, _, _ = aug.random_distortion(img)
    # resize, normalize & append
    x_out.append(cv2.resize(distorted, (200, 66))  / 255.0)
    y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
  train_batch_pointer += batch_size
  return x_out, y_out

def LoadValBatch(batch_size):
  global val_batch_pointer
  x_out = []
  y_out = []
  for i in range(0, batch_size):
    x_out.append(cv2.resize(cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images]), (200, 66)) / 255.0)
    y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
  val_batch_pointer += batch_size
  return x_out, y_out