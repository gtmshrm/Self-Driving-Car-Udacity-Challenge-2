import cv2
import math
import matplotlib.pyplot as plt
import numpy as np


def get_horizon_y(img, draw=False, min_y=200, max_y=300, hough_threshold=150):
  ''' Estimate horizon y coordinate using Canny edge detector and Hough transform. '''

  gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

  if draw:
    fig = plt.figure()
    plt.imshow(gray, cmap='gray')

  edges = cv2.Canny(gray,20,150,apertureSize = 3)

  if draw:
    fig = plt.figure()
    plt.imshow(edges, cmap='gray')

  lines = None
  horizon = None
  horizon_y = 1000

  while lines is None or horizon is None:

    lines = cv2.HoughLines(edges,1,np.pi/180, hough_threshold)

    if lines is None:
      hough_threshold = hough_threshold - 10
      continue

    horizontal_lines = []

    for i, line in enumerate(lines):
      for rho,theta in line:

        # just the horizontal lines
        if np.sin(theta) > 0.9999:

          if rho < horizon_y and rho >= min_y and rho <= max_y:
            horizon_y = rho
            horizon = line

    if horizon is None:
      hough_threshold = hough_threshold - 10

  if draw and not horizon is None:

    for rho,theta in horizon:
      a = np.cos(theta)
      b = np.sin(theta)

      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))

      cv2.line(gray,(x1,y1),(x2,y2),(255,255,255),2)

    fig = plt.figure()
    plt.imshow(gray, cmap='gray')

  if horizon is None:
    print('Horizon not found. Return default estimate.')
    return min_y

  return int(horizon_y)


def eulerToRotation(theta):
  ''' Calculates Rotation Matrix given euler angles. '''

  R_x = np.array([
          [1,         0,                  0                   ],
          [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
          [0,         math.sin(theta[0]), math.cos(theta[0])  ]
          ])

  R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
          [0,                     1,      0                   ],
          [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
          ])

  R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
          [math.sin(theta[2]),    math.cos(theta[2]),     0],
          [0,                     0,                      1]
          ])


  R = np.dot(R_z, np.dot( R_y, R_x ))

  return R


def translation(t):
  ''' Returns a 2-dimension translation matrix '''

  T = np.array([[1, 0, t[0]],
          [0, 1, t[1]],
          [0, 0, 1]])
  return T


def apply_distortion(img, rotation, shift, pixel_per_meter=160,
           crop_x=10, crop_y=240, draw=False):

  '''
  Applies shift and rotation distortion to image, assuming all points below the
  horizon are on flat ground and all points above the horizon are infinitely far away.
  The distorted image is also cropped to match the proportions used in "End to End Learning for Self-Driving Cars".
  Parameters:
  img - source image
  rotation - 'yaw' rotation angle in radians.
  shift - shift in meters.
  rotation_mean - rotation distribution mean
  rotation_std - rotation distribution standard deviation
  shift_mean - shift distribution mean
  shift_std - shift distribution standard deviation
  crop_x - number of pixels to be cropped from each side of the distorted image.
  crop_y - number of pixels to be cropped from the upper portion of the distorted image.
  crop - enables/disables cropping
  draw - enables/disables drawing using matplotlib (useful for debugging)
  '''

  # convert to pixels
  shift = shift * pixel_per_meter

  #copy = img.copy()

  if draw:
    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(3, 2, 1, title="Original")
    plt.imshow(img)

  #horizon_y = get_horizon_y(img)
  horizon_y = crop_y
  below_horizon = img[crop_y:,:]

  pts = np.array([[0, 0], [below_horizon.shape[1]-1, 0], [below_horizon.shape[1]-1, below_horizon.shape[0]-1],
          [0, below_horizon.shape[0]-1]], dtype=np.float32)

  birds_eye_magic_number = 20

  dst = np.array([
      [0, 0],
      [below_horizon.shape[1] - 1, 0],
      [below_horizon.shape[1] - 1, (below_horizon.shape[0] * birds_eye_magic_number) - 1],
      [0, (below_horizon.shape[0] * birds_eye_magic_number) - 1]], dtype=np.float32)

  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(pts, dst)
  '''below_horizon = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0] * birds_eye_magic_number))
  if draw:
    fig.add_subplot(3, 2, 2, title="Bird's eye view")
    plt.imshow(below_horizon)'''

  T1 = translation([-(below_horizon.shape[1]/2 + shift), -(below_horizon.shape[0] * birds_eye_magic_number)])
  T2 = translation([below_horizon.shape[1]/2, below_horizon.shape[0] * birds_eye_magic_number])
  T = np.dot(T2, np.dot(eulerToRotation([0., 0., rotation]), T1))
  '''warped = cv2.warpPerspective(below_horizon, T, (below_horizon.shape[1], below_horizon.shape[0]))
  if draw:
    fig.add_subplot(3, 2, 3, title="rotation: {:.2f}; shift: {:.2f}".format(rotation, shift))
    plt.imshow(warped)'''
  T = np.dot(T, M)
  T = np.dot(np.linalg.inv(M), T)

  warped = cv2.warpPerspective(below_horizon, T, (below_horizon.shape[1], below_horizon.shape[0]))

  '''if draw:
    fig.add_subplot(3, 2, 4, title="Inverse warp transform")
    plt.imshow(warped)'''

  #copy[horizon_y:] = warped[horizon_y:] * (warped[horizon_y:] > 0) + copy[horizon_y:] * (1 - (warped[horizon_y:] > 0))
  #copy[horizon_y:] = warped[horizon_y:]
  #copy[horizon_y - 3: horizon_y + 3] = cv2.blur(copy[horizon_y - 3: horizon_y + 3], (1,5))

  #if crop:
  img = warped[:, crop_x:img.shape[1]-crop_x]

  if draw:
    fig.add_subplot(3, 2, 2, title="Final result after cropping")
    plt.imshow(img)

  return img


def random_distortion(image, rotation=None, shift=None, rotation_mean=0, rotation_std=0.002,
            shift_mean=0, shift_std=0.1):
  '''
  Applies random shift and rotation distortion to image.
  Parameters:
  img - source image
  rotation - 'yaw' rotation angle in radians. If None, value is sampled from normal distribution.
  shift - shift in meters. If None, value is sampled from normal distribution.
  rotation_mean - rotation distribution mean
  rotation_std - rotation distribution standard deviation
  shift_mean - shift distribution mean
  shift_std - shift distribution standard deviation
  '''

  if rotation is None:
    rotation = np.random.normal(rotation_mean, rotation_std)

  if shift is None:
    shift = np.random.normal(shift_mean, shift_std)

  return apply_distortion(image, rotation, shift), rotation, shift


def get_steer_back_angle(steering_wheel_angle, speed, rotation, shift, steer_back_time = 2., fps = 20,
           wheel_base = 2.84988, steering_ratio = 14.8):

  dt = (1./fps)
  #shift0 = shift
  #rotation0 = rotation
  theta = math.pi/2. + rotation
  # true vehicle velocity
  v = speed
  vx = math.cos(theta) * v

  # assume constant acceleration
  ax = (-shift - vx * steer_back_time) * 2. / (steer_back_time * steer_back_time)

  # calculate velocity x and shift after dt
  vx = vx + ax * dt
  shift = shift + vx * dt + ax * dt * dt / 2.

  # steer back angular velocity
  vtheta = (math.acos(vx / v) - theta) / dt

  # calculate theta after dt
  #theta = theta + vtheta * dt
  theta = math.acos(vx / v)

  # true angular velocity
  vtheta_truth = math.tan(steering_wheel_angle / steering_ratio) * v / wheel_base

  #print(vtheta, vtheta_truth, left_steering.iloc[i].steering_wheel_angle)

  # we have two add "steer back" and true angular velocities to calculate final steering angle
  vtheta = vtheta + vtheta_truth

  wheel_angle = math.atan(vtheta * wheel_base / v)
  steering_wheel_angle = wheel_angle * steering_ratio

  rotation = -(math.pi/2. - theta)
  return rotation, shift, steering_wheel_angle



def steer_back_distortion(image, steering_wheel_angle, speed, rotation=None, shift=None,
              initial_rotation=0, initial_shift=0):
  ''' Utility function to easily generate new labeled images with random rotation and shift. '''

  distorted, rotation, shift = random_distortion(image, rotation=rotation, shift=shift)
  rotation, shift, steering_wheel_angle = get_steer_back_angle(steering_wheel_angle, speed, rotation + initial_rotation, shift + initial_shift)

  return distorted, steering_wheel_angle, rotation, shift
