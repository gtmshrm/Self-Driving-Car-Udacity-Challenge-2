'''
To view tensorboard train and validation loss
In the 1st terminal, run the command line:
  --> tensorboard --logdir=./train_logs
  Then open http://0.0.0.0:6006/ into your web browser
In the 2nd terminal, run the command line:
  --> tensorboard --logdir=./val_logs --port=8008"
  Then open http://0.0.0.0:8008/ into your web browser
'''
import os
import os.path
import tensorflow as tf
import driving_data
import model
import numpy as np

LOGDIR = './save'
CKPT_FILE = './save/model.ckpt'
TRAIN_TENSORBOARD_LOG = './train_logs'
VAL_TENSORBOARD_LOG = './val_logs'

sess = tf.InteractiveSession()

loss = tf.reduce_mean(tf.square(tf.sub(model.y_, model.y)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

saver = tf.train.Saver()

# create a summary to monitor cost tensor
train_summary = tf.scalar_summary("train_loss", loss)
val_summary = tf.scalar_summary("val_loss", loss)

if not os.path.exists(LOGDIR):
  os.makedirs(LOGDIR)

if os.path.isfile(CKPT_FILE):
  saver.restore(sess, CKPT_FILE)
else:
  sess.run(tf.initialize_all_variables())

if not os.path.exists(TRAIN_TENSORBOARD_LOG):
  os.makedirs(TRAIN_TENSORBOARD_LOG)
if not os.path.exists(VAL_TENSORBOARD_LOG):
  os.makedirs(VAL_TENSORBOARD_LOG)

# op to write logs to Tensorboard
train_summary_writer = tf.train.SummaryWriter(TRAIN_TENSORBOARD_LOG, graph=tf.get_default_graph())
val_summary_writer = tf.train.SummaryWriter(VAL_TENSORBOARD_LOG, graph=tf.get_default_graph())

batch_size = 100

for i in range(int(driving_data.num_images * 3)):
  xs_train, ys_train = driving_data.LoadTrainBatch(batch_size)
  train_step.run(feed_dict={model.x: xs_train, model.y_: ys_train, model.keep_prob: 0.8})
  
  if i % 10 == 0:
    xs_val, ys_val = driving_data.LoadValBatch(batch_size)
    # write logs at every iteration
    train_loss = train_summary.eval(feed_dict={model.x:xs_train, model.y_: ys_train, model.keep_prob: 1.0})
    val_loss = val_summary.eval(feed_dict={model.x:xs_val, model.y_: ys_val, model.keep_prob: 1.0})
    train_summary_writer.add_summary(train_loss, i)
    val_summary_writer.add_summary(val_loss, i)
    train_loss = loss.eval(feed_dict={model.x:xs_train, model.y_: ys_train, model.keep_prob: 1.0})
    val_loss = loss.eval(feed_dict={model.x:xs_val, model.y_: ys_val, model.keep_prob: 1.0})
    print("step: %d, loss: %g, val loss: %g" % (i, train_loss, val_loss))

  if i % 100 == 0:
    checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
    filename = saver.save(sess, checkpoint_path)
print("Model saved in file: %s" % filename)
