import tensorflow as tf
import scipy

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

x = tf.placeholder(tf.float32, shape=[None, 120, 160, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x

# first convolutional layer
W_conv1 = weight_variable([5, 5, 3, 24])
b_conv1 = bias_variable([24])

h_conv1m = conv2d(x_image, W_conv1, 2) + b_conv1
h_conv1 = tf.maximum(0.01*h_conv1m, h_conv1m)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 24, 36])
b_conv2 = bias_variable([36])

h_conv2m = conv2d(h_conv1, W_conv2, 2) + b_conv2
h_conv2 = tf.maximum(0.01*h_conv2m, h_conv2m)

# third convolutional layer
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable([48])

h_conv3m = conv2d(h_conv2, W_conv3, 2) + b_conv3
h_conv3 = tf.maximum(0.01*h_conv3m, h_conv3m)

# fourth convolutional layer
W_conv4 = weight_variable([5, 5, 48, 64])
b_conv4 = bias_variable([64])

h_conv4m = conv2d(h_conv3, W_conv4, 1) + b_conv4
h_conv4 = tf.maximum(0.01*h_conv4m, h_conv4m)

# fifth convolutional layer
W_conv5 = weight_variable([8, 8, 64, 64])
b_conv5 = bias_variable([64])

h_conv5m = conv2d(h_conv4, W_conv5, 1) + b_conv5
h_conv5 = tf.maximum(0.01*h_conv5m, h_conv5m)


# FCL 1
W_fc1 = weight_variable([384, 1164])
b_fc1 = bias_variable([1164])

h_conv5_flat = tf.reshape(h_conv5, [-1, 384])

h_fc1m = tf.matmul(h_conv5_flat, W_fc1) + b_fc1
h_fc1 = tf.maximum(0.01*h_fc1m, h_fc1m)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# FCL 2
W_fc2 = weight_variable([1164, 100])
b_fc2 = bias_variable([100])

h_fc2m = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
h_fc2 = tf.maximum(0.01*h_fc2m, h_fc2m )

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# FCL 3
W_fc3 = weight_variable([100, 50])
b_fc3 = bias_variable([50])

h_fc3m = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
h_fc3 = tf.maximum(0.01*h_fc3m, h_fc3m )


h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# FCL 3
W_fc4 = weight_variable([50, 10])
b_fc4 = bias_variable([10])

h_fc4m = tf.matmul(h_fc3_drop, W_fc4) + b_fc4
h_fc4 = tf.maximum(0.01*h_fc4m, h_fc4m )


h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

# Output
W_fc5 = weight_variable([10, 1])
b_fc5 = bias_variable([1])

y = tf.mul(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) # scale the atan output by 2
