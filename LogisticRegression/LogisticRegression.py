import os
import numpy as np
import tensorflow as tf

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
    if action == 0: # decrease 3%
      self.adaptive_lr *= 0.97
    elif action == 1: # reset
      self.adaptive_lr = self.config['learning_rate']

    train_x, train_y = self.dataloader.next_batch(self.config['sample_ind'], \
                                                  self.config['batch_size'])
    _, c = self.session.run([self.optimizer, self.cost], feed_dict={ \
            self.x:train_x, self.y:train_y, self.adaptive_lr_holder:self.adaptive_lr})
    return c

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
  'n_dim' : 10,
  'n_train_samples' : 90,
  'n_test_samples' : 100,
  'sample_ind' : 10,

  # Parameters
  'learning_rate' : 1e-2,
  'n_epochs' : 50,
  'batch_size' : 5,
  'display_step' : 10000
  }

  lr = LogisticRegression(config);

  #print "Init accuracy:", lr.test()

  for epoch in xrange(config['n_epochs']):
    avg_cost = 0
    batch_ind = 0
    n_batches = int(config['n_train_samples']/config['batch_size'])
    for i in xrange(n_batches):
      action = np.array([1, 0])
      c = lr.step(action)
      avg_cost += c / n_batches
      batch_ind += 1;
      if (batch_ind) % config['display_step'] == 0:
        print '[%02d/%02d]'%(epoch+1, batch_ind), \
              "cost=", "{:.9f}".format(c)#, "test accuracy:", lr.test()
    print '[%02d]'%(epoch+1), \
      "cost=", "{:.9f}".format(avg_cost)#, "test accuracy:", lr.test()

  print "Training Finished"

  print "Test Accuracy:", lr.test()
