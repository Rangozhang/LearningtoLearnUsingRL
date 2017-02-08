import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

class SoftmaxRegression(object):
  def __init__(self, config):
    self.config = config

    # adaptive parameters: lr
    self.adaptive_lr = self.config['learning_rate']

    self.adaptive_lr_holder = tf.placeholder(tf.float32, shape=[])
    self.x = tf.placeholder(tf.float32, [None, mnist.IMAGE_PIXELS])
    self.y = tf.placeholder(tf.float32, [None, mnist.NUM_CLASSES])
    self.W = tf.Variable(tf.zeros([mnist.IMAGE_PIXELS, mnist.NUM_CLASSES]))
    self.b = tf.Variable(tf.zeros([mnist.NUM_CLASSES]))

    self.pred = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
    self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), \
            reduction_indices=1))

    self.optimizer = tf.train.GradientDescentOptimizer( \
            learning_rate=self.adaptive_lr_holder).minimize(self.cost)

    self.dataloader = input_data.read_data_sets("/home/users/yu01.zhang/dataset/MNIST_data", one_hot=True)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

  def step(self, action):
    # define action
    action = np.argmax(action)
    if action == 0: # decrease 1%
      self.adaptive_lr *= 0.97
    elif action == 1: # increase 1%
      self.adaptive_lr *= 1.001
    elif action == 2: # stay the same
      self.adaptive_lr = self.adaptive_lr

    train_x, train_y = self.dataloader.train.next_batch(self.config['batch_size'])
    _, c = self.session.run([self.optimizer, self.cost], feed_dict={ \
            self.x:train_x, self.y:train_y, self.adaptive_lr_holder:self.adaptive_lr})
    return c

  def print_lr(self):
    return self.adaptive_lr

  def test(self):
    correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_val = self.session.run([accuracy], feed_dict={self.x:self.dataloader.test.images, \
                   self.y:self.dataloader.test.labels})[0]
    return accuracy_val

if __name__ == "__main__":
  config = {
  'n_training_samples' : 55000,
  'learning_rate' : 5e-1,
  'n_epochs' : 200,
  'batch_size' : 100,
  'display_step' : np.Inf,
  'decay': 0
  }
  print config

  lr = SoftmaxRegression(config);

  #print "Init accuracy:", lr.test()

  plt_y = []
  n_steps = 0
  for epoch in xrange(config['n_epochs']):
    avg_cost = 0
    batch_ind = 0
    n_batches = int(config['n_training_samples']/config['batch_size'])
    for batch in xrange(n_batches):
      decay = config['decay']
      if decay == 1:
        if epoch > 100 and batch == 0:
          action = np.array([1, 0, 0])
        else:
          action = np.array([0, 0, 1])
      elif decay == 0:
        action = np.array([0, 0, 1])
      elif decay == -1:
        action = np.array([0, 1, 0])
      c = lr.step(action)
      avg_cost += c / n_batches
      batch_ind += 1
      n_steps += 1
      if (batch_ind) % config['display_step'] == 0:
        print '[%02d/%02d]'%(epoch+1, batch_ind), \
              "cost=", "{:.9f}".format(c)#, "test accuracy:", lr.test()
    print '[%02d]'%(epoch+1), \
            "cost=", "{:.9f}".format(avg_cost), "lr=", "{:.5f}".format(lr.print_lr()) #, "test accuracy:", lr.test()
    plt_y.append(avg_cost)

  print "Training Finished"
  print "Test Accuracy:", lr.test()

  plt_x = np.arange(config['n_epochs'])
  plt_y = np.asarray(plt_y)
  plt.plot(plt_x, plt_y, 'r')
  axes = plt.gca()
  axes.set_ylim([0.2,0.4])
  plt.show()

