import os
import time
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

    self.logits = tf.matmul(self.x, self.W) + self.b
    self.pred = tf.nn.softmax(self.logits)
    # self.cost_batch = -tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1)
    # self.cost = tf.reduce_mean(self.cost_batch)
    # handle cross entropy better numerically
    self.cost_batch = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y)
    self.cost = tf.reduce_mean(self.cost_batch)

    # self.optimizer = tf.train.GradientDescentOptimizer( \
    self.optimizer = tf.train.AdamOptimizer( \
            learning_rate=self.adaptive_lr_holder)
    self.grads = self.optimizer.compute_gradients(self.cost, var_list=self.trainable_var)
    self.apply_grads = self.optimizer.apply_gradients(self.grads)

    # self.var_grad = tf.gradients(self.cost, [self.W, self.b])

    self.dataloader = input_data.read_data_sets("/home/users/yu01.zhang/dataset/MNIST_data", one_hot=True)
    config_proto = tf.ConfigProto(log_device_placement=False)
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.2
    # config_proto.gpu_options.allow_growth = True
    # config_proto.gpu_options.visible_device_list = '3'
    self.session = tf.Session(config=config_proto)
    self.session.run(tf.global_variables_initializer())

    self.test_x,  self.test_y  = self.dataloader.test.images, self.dataloader.test.labels

  def reset(self):
    self.adaptive_lr = self.config['learning_rate']
    self.costgrad_history = deque([0]*10)
    self.n_step = 0
    self.reward_sum = 0
    self.session.run(tf.global_variables_initializer())

  def step(self, action):
    # define action
    action = np.argmax(action)

    if action == 0: # decrease 3%
      self.adaptive_lr *= 0.97
    elif action == 1: # reset
      self.adaptive_lr = self.adaptive_lr

    train_x, train_y = self.dataloader.train.next_batch(self.config['batch_size'])

    test_c = self.session.run([self.cost], feed_dict={self.x:self.test_x, self.y:self.test_y})[0]

    c, c_batch, grads, var, _ = self.session.run([self.cost, self.cost_batch, \
            [grad[0] for grad in self.grads], self.trainable_var, self.apply_grads], \
            feed_dict={self.x:train_x, self.y:train_y, self.adaptive_lr_holder:self.adaptive_lr})

    # post_c_batch, post_var = self.session.run([self.cost_batch, self.trainable_var], \
    #                                            feed_dict={self.x:train_x, self.y:train_y})

    test_post_c = self.session.run([self.cost], feed_dict={self.x:self.test_x, self.y:self.test_y})[0]

    # self.costgrad_history.append(test_post_c-test_c)
    # self.costgrad_history.popleft()

    # ############## 1. state ##############
    # delta_c_batch = post_c_batch - c_batch
    # delta_c_batch_stats = get_stats(delta_c_batch)
    # delta_var = flatten_list(post_var) - flatten_list(var) # suppose to be amount to -lr*grads if no momentum
    # delta_var_stats = get_stats(delta_var)
    # # grads_flatten = flatten_list(grads)
    # # grads_flatten_stats = get_stats(grads_flatten)

    # (135,)
    state_list = np.hstack([# # delta_c_batch,
                            # delta_c_batch_stats,
                            # # delta_var,
                            # delta_var_stats,
                            # # grads_flatten,
                            # # grads_flatten_stats,
                            # np.asarray(self.costgrad_history),
                            self.adaptive_lr])

    ############## 2. terminal ##############
    self.n_step += 1
    if self.n_step >= self.config['n_epochs'] * self.config['n_batches']:
      terminal = True
      self.n_step = 0
    else:
      terminal = False


    ############## 3. reward ##############
    # -(np.log(post_c) - np.log(c))
    # log_diff = c - post_c if self.n_step != 0 else 0
    log_diff = np.log(test_c) - np.log(test_post_c) if self.n_step != 0 else 0
    # self.reward_sum += log_diff

    # op.1
    # reward = self.reward_sum / self.n_step

    # op.2
    reward = log_diff

    # op.3: need to backprop by all actions?
    # reward = self.reward_sum / self.n_step if terminal else 0

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

  cost_holder = tf.placeholder(tf.float32)
  tf.summary.scalar('cost', cost_holder)
  merged = tf.summary.merge_all()

  config = {
  'n_training_samples' : 55000,
  'learning_rate' : 4e-3,
  'decay': 1,
  'n_epochs' : 50,
  'n_batches' : 550, #config['n_training_samples']/config['batch_size']
  'batch_size' : 100,
  'display_step' : 110
  }
  print config

  lr = SoftmaxRegression(config);
  #writer = tf.train.SummaryWriter("./res/fig/{:10d}".format(int(time.time())))
  writer = tf.summary.FileWriter("./res/env_fig/{:.5f}_{:d}_{:10d}".format(config['learning_rate'], \
                                                                    config['decay'], \
                                                                    int(time.time())))

  #print "Init accuracy:", lr.test()

  #plt_y = []
  terminal = False
  batch, epoch, avg_cost, avg_max_grad, avg_min_grad, avg_reward = 0, 0, 0, 0, 0, 0
  while not terminal:
    batch_ind = lr.get_n_step()
    if batch % config['n_batches'] == 0:
        batch = 0

    decay = config['decay']
    if decay == 1:
      if epoch > 20 and batch == 0:
        action = np.array([1, 0])
      else:
        action = np.array([0, 1])
    elif decay == 0:
      action = np.array([0, 1])

    c, grads, state, reward, terminal = lr.step(action)
    avg_cost += c / config['n_batches']
    avg_reward += reward / config['n_batches']
    grads_val_max = np.asarray([grad.max() for grad in grads]).max()
    grads_val_min = np.asarray([grad.min() for grad in grads]).min()
    avg_max_grad += grads_val_max / config['n_batches']
    avg_min_grad += grads_val_min / config['n_batches']
    if batch_ind % config['display_step'] == 0:
      print '[%02d/%03d]'%(epoch+1, batch_ind), \
            "cost={:.9f}".format(c), "lr={:.5f}".format(lr.print_lr()), \
          "avg_grad=({:.5f},{:.5f})".format(grads_val_max, grads_val_min), \
          "state.shape=({:d}) reward={:.5f} terminal={}".format(state.shape[0], reward, terminal)
    if batch_ind % config['n_batches'] == config['n_batches'] - 1:
      print '[%02d]'%(epoch+1), \
              "avg_cost=", "{:.9f}".format(avg_cost), "lr=", "{:.5f}".format(lr.print_lr()), \
              "avg_grad=", "({:.5f},{:.5f})".format(avg_max_grad, avg_min_grad), \
              "avg_r={:.5f}".format(avg_reward)
      res_merged = lr.session.run([merged], feed_dict={cost_holder: avg_cost})[0]
      writer.add_summary(res_merged, epoch)

      avg_cost, avg_max_grad, avg_min_grad, avg_reward = 0, 0, 0, 0
      epoch += 1

    batch += 1
    #plt_y.append(avg_cost)

  print "Training Finished"
  print "Test Accuracy:", lr.test()

  # plt_x = np.arange(config['n_epochs'])
  # plt_y = np.asarray(plt_y)
  # plt.plot(plt_x, plt_y, 'r')
  # axes = plt.gca()
  # axes.set_ylim([0.2,0.4])
  # plt.show()

