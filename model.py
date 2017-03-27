import numpy as np
import tensorflow as tf


def flatten(x_tensor):
    flat_dim = np.prod(x_tensor.get_shape().as_list()[1:])
    return tf.reshape(x_tensor, shape=(-1,flat_dim))

def fully_conn(x_tensor, num_outputs):
    weights = tf.Variable(tf.truncated_normal(
                    [x_tensor.get_shape().as_list()[1],num_outputs],
                    stddev=0.1))
    biases = tf.Variable(tf.constant(0.0, shape=[num_outputs]))
    return tf.nn.relu(tf.matmul(x_tensor,weights) + biases)

def output(x_tensor, num_outputs):
    weights = tf.Variable(tf.truncated_normal(
                    [x_tensor.get_shape().as_list()[1],num_outputs],
                    stddev=0.1))
    biases = tf.Variable(tf.constant(0.0, shape=[num_outputs]))
    return tf.matmul(x_tensor,weights) + biases

def conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides):
    weights = tf.Variable(tf.truncated_normal(
                    [conv_ksize[0],conv_ksize[1],x_tensor.get_shape().as_list()[3],conv_num_outputs],
                    stddev=0.1))
    biases = tf.Variable(tf.constant(0.0, shape=[conv_num_outputs]))
    x = tf.nn.conv2d(x_tensor, weights, [1,conv_strides[0],conv_strides[1],1], padding='SAME')
    return tf.nn.relu(x+biases)

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    weights = tf.Variable(tf.truncated_normal(
                    [conv_ksize[0],conv_ksize[1],x_tensor.get_shape().as_list()[3],conv_num_outputs],
                    stddev=0.1))
    biases = tf.Variable(tf.constant(0.0, shape=[conv_num_outputs]))
    x = tf.nn.conv2d(x_tensor, weights, [1,conv_strides[0],conv_strides[1],1], padding='SAME')
    x = tf.nn.relu(x+biases)

    x = tf.nn.max_pool(x, [1, pool_ksize[0], pool_ksize[1], 1], [1, pool_strides[0], pool_strides[1], 1], padding='SAME')
    return x

# Placeholders for model inputs, outputs and dropout rates
x = tf.placeholder(tf.float32, shape=[None,66,200,3])
y_ = tf.placeholder(tf.float32, shape=[None,1])

keep_prob = tf.placeholder(tf.float32)

# Model architecture
conv = conv2d(x, 3, (1,1), (1,1))

conv = conv2d(conv, 32, (3,3), (1,1))
conv = conv2d_maxpool(conv, 32, (3,3), (1,1), (2,2), (2,2))

conv = conv2d(conv, 64, (3,3), (1,1))
conv = conv2d_maxpool(conv, 64, (3,3), (1,1), (2,2), (2,2))

conv = conv2d(conv, 128, (3, 3), (1,1))
conv = conv2d_maxpool(conv, 128, (3,3), (1,1), (2,2), (2,2))
conv = tf.nn.dropout(conv, keep_prob)

flat = flatten(conv)

dense = fully_conn(flat, 512)
dense = tf.nn.dropout(dense, keep_prob)
dense = fully_conn(dense, 64)

y = tf.nn.tanh(output(dense, 10))
