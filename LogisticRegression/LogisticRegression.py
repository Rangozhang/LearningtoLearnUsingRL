import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.dataloader import DataLoader

class LogisticRegression(object):
  def __init__(self, config):
    self.config = config

    # adaptive parameters: lr
    self.adaptive_lr = self.config['learning_rate']

    self.adaptive_lr_holder = tf.placeholder(tf.float32, shape=[])
    self.x = tf.placeholder(tf.float32, [None, self.config['n_dim']])
    self.y = tf.placeholder(tf.float32, [None, 2])
    self.W = tf.Variable(tf.zeros([self.config['n_dim'], 2]))
    self.b = tf.Variable(tf.zeros([2]))

    self.pred = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
    self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), \
            reduction_indices=1))

    self.optimizer = tf.train.GradientDescentOptimizer( \
            learning_rate=self.adaptive_lr_holder).minimize(self.cost)

    self.dataloader = DataLoader(self.config['data_dir'], 0.8)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

  def step(self, action):
    # define action
    action = np.argmax(action)
    if action == 0: # decrease 1%
      self.adaptive_lr *= 0.99
    elif action == 1: # increase 1%
      self.adaptive_lr *= 1.001
    elif action == 2: # reset
      self.adaptive_lr = self.config['learning_rate']

    train_x, train_y = self.dataloader.next_batch(self.config['sample_ind'], \
                                                  self.config['batch_size'])
    _, c = self.session.run([self.optimizer, self.cost], feed_dict={ \
            self.x:train_x, self.y:train_y, self.adaptive_lr_holder:self.adaptive_lr})
    return c

  def print_lr(self):
    return self.adaptive_lr

  def test(self):
    correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = self.dataloader.next_batch(self.config['sample_ind'], \
            self.config['n_test_samples'], isTraining=False, isTrainingOpt=True)
    accuracy_val = self.session.run([accuracy], feed_dict={self.x:test_x, self.y:test_y})[0]
    return accuracy_val

if __name__ == "__main__":
  config = {
  # Config
  'data_dir' : 'dataset',
  'n_dim' : 3,
  'n_train_samples' : 90,
  'n_test_samples' : 100,
  'n_instances_per_sample': 100,
  'sample_ind' : 0,

  # Parameters
  'learning_rate' : 1e-1,
  'n_epochs' : 1000,
  'batch_size' : 100,
  'display_step' : np.Inf
  }

  lr = LogisticRegression(config);

  #print "Init accuracy:", lr.test()

  plt_y = []
  n_steps = 0
  for epoch in xrange(config['n_epochs']):
    avg_cost = 0
    batch_ind = 0
    n_batches = int(config['n_instances_per_sample']/config['batch_size'])
    for i in xrange(n_batches):
      decay = -1
      if decay == 1:
        #if epoch > 80:
        #  action = np.array([1, 0, 0])
        #else:
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

  plt_x = np.arange(config['n_epochs'])
  plt_y = np.asarray(plt_y)
  plt.plot(plt_x, plt_y, 'r')
  plt.show()

  print "Test Accuracy:", lr.test()
