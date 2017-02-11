import numpy as np
import tensorflow as tf
import random
import argparse
from collections import deque

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_actions', 2, 'number of actions')
tf.app.flags.DEFINE_integer('num_episodes', 100, 'number of episodes')
tf.app.flags.DEFINE_integer('len_episodes', 550, 'length of an episode')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('observ_dim', 200, 'dimension of state')
tf.app.flags.DEFINE_integer('state_num', 4, 'number of observ are stacked in a state')
tf.app.flags.DEFINE_float('gamma', 0.99, 'discount factor on past Q values')
tf.app.flags.DEFINE_float('start_e', 1.0, 'chance of random action at the beginning')
tf.app.flags.DEFINE_float('end_e', 0.1, 'chance of random action at the end')
tf.app.flags.DEFINE_float('lr', 1e-3, 'learning rate')
tf.app.flags.DEFINE_boolean('isTraining', True, 'is training?')

# uncertain
tf.app.flags.DEFINE_integer('explore', 10000, 'observe before training')
tf.app.flags.DEFINE_integer('observe', 64, 'observe before training') # as long as > batch_size
tf.app.flags.DEFINE_float('tau', 1e-3, 'rate to update target network towards primary network')
tf.app.flags.DEFINE_integer('memory_size', 2000, 'replay memory size')
tf.app.flags.DEFINE_float('training_freq', 2, 'How ofen to perform a trianing step')

def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))
    out = tf.nn.bias_add(tf.matmul(input_, w), b)
    if activation_fn != None:
      return activation_fn(out), w, b
    else:
      return out, w, b

class memory_buffer(object):
  def __init__(self, buffer_size):
    self.size = buffer_size
    # mem_buffer: (s, a, r, s1, t)
    self.mem_buffer = deque()
  def add(self, experience):
    self.mem_buffer.append(experience)
    if len(self.mem_buffer) > self.size:
      self.mem_buffer.popleft()
  def sample(self, batch_size):
    return random.sample(self.mem_buffer, batch_size)

class Qnetwork(object):
  def __init__(self):
    self.memory = memory_buffer(FLAGS.memory_size)
    self.step = 0
    self.batch_size = FLAGS.batch_size
    self.epsilon = FLAGS.start_e
    self.sess = tf.Session()
    with tf.variable_scope('primary'):
      self.state_input, self.Q, self.w = self.create_model()
    with tf.variable_scope('target'):
      self.state_input_t, self.Q_t, self.w_t = self.create_model()
    with tf.variable_scope('target_update'):
      self.w_t_input = {}
      self.w_t_update_op = {}
      for name in self.w_t.keys():
        self.w_t_update_op[name] = self.w_t[name].assign(tau*self.w[name], (1-tau)*self.w_t[name])
        #self.w_t_update_op[name] = self.w_t[name].assign(tau*self.w[name].value(), (1-tau)*self.w_t[name].value())
        #self.w_t_input[name] = tf.placeholder('float32', self.w_t[name].get_shape().as_list(), name=name)
        #self.w_t_update_op[name] = self.w_t[name].assign(self.w_t_input[name])
    self.action_pred = argmax(self.Q, 1)
    self.create_train_op()

  def create_model(self):
    W = {}
    state_input = tf.placeholder('float32', [None, FLAGS.observ_dim*FLAGS.state_num])
    l1, W['l1_w'], W['l1_b'] = linear(state_input, 512, \
            activation_fn=tf.nn.relu, name='l1')
    l2, W['l2_w'], W['l2_b'] = linear(l1, 512, \
            activation_fn=tf.nn.relu, name='l2')
    v_hid, W['v_hid_w'], W['v_hid_b'] = linear(l2, 256, \
            activation_fn=tf.nn.relu, name='v_hid')
    adv_hid, W['adv_hid_w'], W['adv_hid_b'] = linear(l2, 256, \
            activation_fn=tf.nn.relu, name='adv_hid')
    v, W['v_w'], W['v_b'] = linear(v_hid, 1, name='v')
    adv, W['adv_w'], w['adv_b'] = linear(adv_hid, FLAGS.num_actions, name='adv')
    Q = v + (adv - tf.reduce_mean(adv, reduction_indices=1, keep_dims=True))
    return state_input, Q, W

  def update_target_network(self):
    for name in self.w.keys():
      self.session.run(self.w_t_update_op[name])
      #self.w_t_update_op[name].eval({self.w_t_input[name]:self.w[name].eval()})

  def create_train_op(self):
    self.Q_gt = tf.placeholder('float32', [None])

    #self.action = tf.placeholder('float32', [None])
    #self.action_onehot = tf.one_hot(self.action, FLAGS.num_actions, dtype=tf.float32)
    self.action = tf.placeholder('float32', [None, FLAGS.num_actions])
    self.Q_pred = tf.reduce_sum(self.action*self.Q, reduction_indices=1)
    self.cost = tf.reduce_mean(tf.square(self.Q_gt-self.Q_pred))
    self.trainer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(self.cost)

  def init_state(self, observation):
    assert (observation.size == 2), "Invalid observation!"
    obvs = []
    for i in xrange(FLAGS.state_num):
      obvs.append(observation)
    self.cur_state = np.concatenate(obvs, axis=1)

  def set_perception(self, action, reward, next_observ, terminal):
    assert (next_observ.size == 2), "Invalid next observation"

    ### 1. update memory
    next_state = np.append(self.cur_state[:, FLAGS.observ_dim:], next_observ, axis=1)
    self.memory.add((self.cur_state, action, reward, next_state, terminal))

    ### 2. train
    if self.step > FLAGS.observe and FLAGS.isTraining:
      self.train()

    ### 3. print info.
    state = ""
    if self.timeStep <= OBSERVE:
        state = "observe"
    elif self.step > FLAGS.observe and self.step <= FLAGS.observe + FLAGS.explore:
        state = "explore"
    else:
        state = "train"

    print "TIMESTEP", self.step, "/ STATE", state, \
          "/ EPSILON", self.epsilon

    ### 4. update training states
    self.cur_state = next_state
    self.step += 1

  def train(self):
    ### 1. obtain training batch from memory
    minibatch = self.memory.sample(self.batch_size)
    state_batch = np.stack([data[0] for data in minibatch])
    action_batch = np.stack([data[1] for data in minibatch])
    reward_batch = np.stack([data[2] for data in minibatch])
    nx_state_batch = np.stack([data[3] for data in minibatch])
    terminal = np.stack([data[4] for data in minibatch])
    assert (state_batch.size == 2 and action_batch.size == 2 and \
            reward_batch.size == 2 and nx_state_batch.size == 2 and \
            terminal.size == 1), "Invalid training input"

    ### 2. calculate Q_gt
    action_pred = self.session.run([self.action_pred], \
            feed_dict={self.state_input: nx_state_batch})
    q_pred_t = self.session.run([self.Q_t], \
            feed_dict={self.state_input_t: nx_state_batch})
    terminal_multiplier = -(terminal-1)
    double_q = q_pred_t[range(FLAGS.batch_size), action_pred]
    Q_gt = reward_batch + FLAGS.gamma * double_q * terminal_multiplier
    self.session.run([self.trainer], feed_dict={
                        self.state_input: state_bach,
                        self.Q_gt: Q_gt,
                        self.action: action_batch})
    self.update_target_network()

  def proc_state(self, state):



if __name__ == "__main__":
  raw_input()
