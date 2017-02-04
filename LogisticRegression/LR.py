import os
import numpy as np
import tensorflow as tf

from dataset.dataloader import DataLoader

# Config
n_dim = 10
n_train_samples = 90
n_test_samples = 100
sample_ind = 0

# Parameters
learning_rate = 0.01
n_epochs = 25
batch_size = 5
display_step = 1

x = tf.placeholder(tf.float32, [None, n_dim])
y = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([n_dim, 2]))
b = tf.Variable(tf.zeros([2]))

pred = tf.nn.softmax(tf.matmul(x, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

dataloader = DataLoader('dataset', 0.8)

with tf.Session() as sess:
  sess.run(init)

  # Training model
  for epoch in xrange(n_epochs):
    avg_cost = 0
    n_batches = int(n_train_samples/batch_size)
    for i in xrange(n_batches):
      train_x, train_y = dataloader.next_batch(sample_ind, batch_size)
      _, c = sess.run([optimizer, cost], feed_dict={x:train_x, y:train_y})
      avg_cost += c / n_batches
    if (epoch+1) % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
  print "Training Finished"

  # Test model
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  test_x, test_y = dataloader.next_batch(sample_ind, n_test_samples, isTraining=False, \
          isTrainingOpt=True)
  print "Accuracy:", accuracy.eval({x: test_x, y: test_y})
  print tf.argmax(pred, 1).eval({x: test_x, y: test_y})
  print tf.argmax(y, 1).eval({x: test_x, y: test_y})
