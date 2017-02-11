import os
import numpy as np
import scipy.stats.stats as st
import tensorflow as tf
from collections import deque
#import matplotlib.pyplot as plt
#matplotlib.use('Agg')

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

def flatten_list(var_list):
    return np.hstack([var.flatten() for var in var_list])

def get_stats(arr):
  return np.asarray([arr.mean(),
                     arr.std(),
                     arr.min(),
                     arr.max(),
                     np.percentile(arr, 25),
                     np.percentile(arr, 50),
                     np.percentile(arr, 75),
                     st.skew(arr),
                     st.kurtosis(arr),
                     st.moment(arr, 1),
                     st.moment(arr, 3),
                     st.moment(arr, 5)])

class SoftmaxRegression(object):
  def __init__(self, config):
    self.config = config

    # adaptive parameters: lr
    self.adaptive_lr = self.config['learning_rate']
    self.costgrad_history = deque([0]*10)
    self.n_step = 0
    self.reward_sum = 0

    self.adaptive_lr_holder = tf.placeholder(tf.float32, shape=[])
    self.x = tf.placeholder(tf.float32, [None, mnist.IMAGE_PIXELS])
    self.y = tf.placeholder(tf.float32, [None, mnist.NUM_CLASSES])

    with tf.variable_scope('model_var'):
      self.W = tf.Variable(tf.zeros([mnist.IMAGE_PIXELS, mnist.NUM_CLASSES]))
      self.b = tf.Variable(tf.zeros([mnist.NUM_CLASSES]))
    self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model_var')

    self.pred = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
    self.cost_batch = -tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1)
    self.cost = tf.reduce_mean(self.cost_batch)

    self.optimizer = tf.train.GradientDescentOptimizer( \
            learning_rate=self.adaptive_lr_holder)
    self.grads = self.optimizer.compute_gradients(self.cost, var_list=self.trainable_var)
    self.apply_grads = self.optimizer.apply_gradients(self.grads)

    # self.var_grad = tf.gradients(self.cost, [self.W, self.b])

    self.dataloader = input_data.read_data_sets("/home/users/yu01.zhang/dataset/MNIST_data", one_hot=True)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

  def step(self, action):
    # define action
    action = np.argmax(action)
    if action == 0: # decrease 1%
      self.adaptive_lr *= 0.97
    elif action == 1: # stay the same
      self.adaptive_lr = self.adaptive_lr
    # elif action == 1: # increase 1%

    train_x, train_y = self.dataloader.train.next_batch(self.config['batch_size'])

    var = self.session.run([self.trainable_var], \
            feed_dict={self.x:train_x, self.y:train_y, self.adaptive_lr_holder:self.adaptive_lr})[0]

    c, c_batch, grads, post_var, _ = self.session.run([self.cost, self.cost_batch, \
            [grad[0] for grad in self.grads], self.trainable_var, self.apply_grads], \
            feed_dict={self.x:train_x, self.y:train_y, self.adaptive_lr_holder:self.adaptive_lr})

    post_c, post_c_batch = self.session.run([self.cost, self.cost_batch], \
            feed_dict={self.x:train_x, self.y:train_y, self.adaptive_lr_holder:self.adaptive_lr})

    self.costgrad_history.append(post_c-c)
    self.costgrad_history.popleft()

    ############## 1. state ##############
    delta_c_batch = post_c_batch - c_batch
    delta_c_batch_stats = get_stats(delta_c_batch)
    delta_var = flatten_list(post_var) - flatten_list(var) # suppose to be amount to -lr*grads if no momentum
    delta_var_stats = get_stats(delta_var)
    grads_flatten = flatten_list(grads)
    grads_flatten_stats = get_stats(grads_flatten)

    # (15838,)
    state_list = np.hstack([delta_c_batch,
                            delta_c_batch_stats,
                            delta_var,
                            delta_var_stats,
                            grads_flatten,
                            grads_flatten_stats,
                            np.asarray(self.costgrad_history),
                            self.adaptive_lr])

    ############## 2. reward ##############
    self.reward_sum += np.log(post_c) - np.log(c) if self.n_step != 0 else 0
    reward = - self.reward_sum / self.n_step if self.n_step != 0 else 0

    self.n_step += 1
    if self.n_step == self.config['n_batches']:
      terminal = True
      self.n_step = 0
    else:
      terminal = False
    return c, grads, state_list, reward, terminal

  def print_lr(self):
    return self.adaptive_lr

  def test(self):
    correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_val = self.session.run([accuracy], feed_dict={self.x:self.dataloader.test.images, \
                   self.y:self.dataloader.test.labels})[0]
    return accuracy_val

  def get_n_step(self):
    return self.n_step

if __name__ == "__main__":
  config = {
  'n_training_samples' : 55000,
  'learning_rate' : 5e-1,
  'n_epochs' : 2,
  'n_batches' : 550, #config['n_training_samples']/config['batch_size']
  'batch_size' : 100,
  'display_step' : 20,
  'decay': 0
  }
  print config

  lr = SoftmaxRegression(config);

  #print "Init accuracy:", lr.test()

  plt_y = []
  for epoch in xrange(config['n_epochs']):
    avg_cost = 0
    avg_max_grad = 0
    avg_min_grad = 0
    terminal = False
    while not terminal:
      decay = config['decay']
      if decay == 1:
        if epoch > 100 and batch == 0:
          action = np.array([1, 0, 0])
        else:
          action = np.array([0, 0, 1])
      elif decay == 0:
        action = np.array([0, 0, 1])
      # elif decay == -1:
      #   action = np.array([0, 1, 0])
      c, grads, state, reward, terminal = lr.step(action)
      avg_cost += c / config['n_batches']
      grads_val_max = np.asarray([grad.max() for grad in grads]).max()
      grads_val_min = np.asarray([grad.min() for grad in grads]).min()
      avg_max_grad += grads_val_max / config['n_batches']
      avg_min_grad += grads_val_min / config['n_batches']
      batch_ind = lr.get_n_step()
      if (batch_ind) % config['display_step'] == 0:
        print '[%02d/%02d]'%(epoch+1, batch_ind), \
              "cost={:.9f}".format(c), "lr={:.5f}".format(lr.print_lr()), \
            "avg_grad=({:.5f},{:.5f})".format(grads_val_max, grads_val_min), \
            "state.shape=({:d}) reward={:.5f} terminal={}".format(state.shape[0], reward, terminal)
    print '[%02d]'%(epoch+1), \
            "cost=", "{:.9f}".format(avg_cost), "lr=", "{:.5f}".format(lr.print_lr()), \
            "avg_grad=", "({:.5f},{:.5f})".format(avg_max_grad, avg_min_grad)
    plt_y.append(avg_cost)

  print "Training Finished"
  print "Test Accuracy:", lr.test()

  # plt_x = np.arange(config['n_epochs'])
  # plt_y = np.asarray(plt_y)
  # plt.plot(plt_x, plt_y, 'r')
  # axes = plt.gca()
  # axes.set_ylim([0.2,0.4])
  # plt.show()

