import numpy as np
import tensorflow as tf
import random
from collections import deque

# len_epoch: 550/action_freq(110) = 5
# len_episode: 50 epoch

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_actions', 2, 'number of actions')
tf.app.flags.DEFINE_integer('num_episodes', 30, 'number of episodes')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('observ_dim', 36, 'dimension of state')
tf.app.flags.DEFINE_integer('state_num', 4, 'number of observ are stacked in a state')
tf.app.flags.DEFINE_float('gamma', 0.99, 'discount factor on past Q values')
tf.app.flags.DEFINE_float('start_e', 1.0, 'chance of random action at the beginning')
tf.app.flags.DEFINE_float('end_e', 0, 'chance of random action at the end')
tf.app.flags.DEFINE_float('lr', 1e-3, 'learning rate')
tf.app.flags.DEFINE_boolean('isTraining', True, 'is training?')
tf.app.flags.DEFINE_integer('explore', 24*50*5, 'observe before training') # 4 episode
tf.app.flags.DEFINE_integer('observe', 1*50*5, 'observe before training') # 1 episode
tf.app.flags.DEFINE_float('tau', 1e-3, 'rate to update target network towards primary network')
tf.app.flags.DEFINE_integer('memory_size', 500, 'replay memory size')
tf.app.flags.DEFINE_integer('memory_sample_freq', 1, 'How often to add a memory')

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
    self.mem_buffer = [[], [], [], [], []]
    self.prob = []
  def add(self, experience, priority=1):
    for i in xrange(5):
      self.mem_buffer[i].append(experience[i])
    self.prob.append(priority)
    if len(self.mem_buffer) > self.size:
      for i in xrange(5):
        self.mem_buffer[i][0:1] = []
      self.prob[0:1] = []
  def sample(self, batch_size):
    prob = np.asarray(self.prob, dtype=float)
    prob = prob / prob.sum()
    inds = np.random.choice(len(self.mem_buffer[0]), batch_size, p=prob, replace=False)
    return [np.asarray(buffer_)[inds] for buffer_ in self.mem_buffer]

class qnet(object):
  def __init__(self):
    self.memory = memory_buffer(FLAGS.memory_size)
    self.step = 0
    self.batch_size = FLAGS.batch_size
    self.epsilon = FLAGS.start_e
    with tf.variable_scope('primary'):
      self.state_input, self.Q, self.w = self.create_model()
    with tf.variable_scope('target'):
      self.state_input_t, self.Q_t, self.w_t = self.create_model()
    with tf.variable_scope('target_update'):
      self.w_t_input = {}
      self.w_t_update_op = {}
      for name in self.w_t.keys():
        self.w_t_update_op[name] = self.w_t[name].assign(FLAGS.tau*self.w[name]+(1-FLAGS.tau)*self.w_t[name])
        #self.w_t_update_op[name] = self.w_t[name].assign(FLAGS.tau*self.w[name].value(), (1-FLAGS.tau)*self.w_t[name].value())
        #self.w_t_input[name] = tf.placeholder('float32', self.w_t[name].get_shape().as_list(), name=name)
        #self.w_t_update_op[name] = self.w_t[name].assign(self.w_t_input[name])
    self.action_pred = tf.argmax(self.Q, 1)
    self.create_train_op()

    config_proto = tf.ConfigProto(log_device_placement=False)
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.4
    # config_proto.gpu_options.allow_growth = True
    # config_proto.gpu_options.visible_device_list = '1'
    self.session = tf.Session(config=config_proto)
    self.session.run(tf.global_variables_initializer())

  def create_model(self):
    W = {}
    state_input = tf.placeholder('float32', [None, FLAGS.observ_dim*FLAGS.state_num])
    l1, W['l1_w'], W['l1_b'] = linear(state_input, 256, \
            activation_fn=tf.nn.relu, name='l1')
    l2, W['l2_w'], W['l2_b'] = linear(l1, 256, \
            activation_fn=tf.nn.relu, name='l2')
    v_hid, W['v_hid_w'], W['v_hid_b'] = linear(l2, 128, \
            activation_fn=tf.nn.relu, name='v_hid')
    adv_hid, W['adv_hid_w'], W['adv_hid_b'] = linear(l2, 128, \
            activation_fn=tf.nn.relu, name='adv_hid')
    v, W['v_w'], W['v_b'] = linear(v_hid, 1, name='v')
    adv, W['adv_w'], W['adv_b'] = linear(adv_hid, FLAGS.num_actions, name='adv')
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
    obvs = []
    for i in xrange(FLAGS.state_num):
      obvs.append(observation)
    self.cur_state = np.hstack(obvs)
    assert (self.cur_state.ndim == 1), "Invalid initial state"

  def set_perception(self, action, reward, next_observ, terminal):
    assert (next_observ.ndim == 1), "Invalid next observation 2 vs. {}".format(next_observ.ndim)

    ### 1. update memory
    next_state = np.append(self.cur_state[FLAGS.observ_dim:], next_observ, axis=0)
    if self.step % FLAGS.memory_sample_freq == 0:
      # compute loss
      # a_p = self.session.run([self.action_pred], \
      #         feed_dict={self.state_input: next_state[np.newaxis, :]})[0][0]
      # q_pt = self.session.run([self.Q_t], \
      #         feed_dict={self.state_input_t: next_state[np.newaxis, :]})[0][0]
      # nx_q = q_pt[a_p]
      # q_gt = reward + FLAGS.gamma * nx_q * (1-terminal)
      # priority = self.session.run([self.cost], feed_dict={self.state_input: self.cur_state[np.newaxis, :],
      #                                                       self.action: action[np.newaxis, :],
      #                                                       self.Q_gt: np.asarray([q_gt])})[0]
      # clip if priority > 1
      # self.memory.add((self.cur_state, action, reward, next_state, terminal), priority if priority < 1 else 1)
      self.memory.add((self.cur_state, action, reward, next_state, terminal))

    ### 2. train
    if self.step > FLAGS.observe and FLAGS.isTraining:
      cost = self.train()
    else:
      cost = 0

    ### 3. print info.
    # state = ""
    # if self.step <= FLAGS.observe:
    #     state = "ob"
    # elif self.step > FLAGS.observe and self.step <= FLAGS.observe + FLAGS.explore:
    #     state = "ex"
    # else:
    #     state = "tr"

    state_str = 'e={:.2f}'.format(self.epsilon)

    ### 4. update training states
    self.cur_state = next_state
    self.step += 1
    return state_str, cost

  def train(self):
    ### 1. obtain training batch from memory
    state_batch, action_batch, reward_batch, nx_state_batch, terminal \
            = self.memory.sample(self.batch_size)
    assert (state_batch.ndim == 2 and action_batch.ndim == 2 and \
            reward_batch.ndim == 1 and nx_state_batch.ndim == 2 and \
            terminal.ndim == 1), "Invalid training input"

    ### 2. calculate Q_gt
    action_pred = self.session.run([self.action_pred], \
            feed_dict={self.state_input: nx_state_batch})[0]
    q_pred_t = self.session.run([self.Q_t], \
            feed_dict={self.state_input_t: nx_state_batch})[0]
    terminal_multiplier = -(terminal-1)
    double_q = q_pred_t[range(FLAGS.batch_size), action_pred]
    Q_gt = reward_batch + FLAGS.gamma * double_q * terminal_multiplier
    c, _ = self.session.run([self.cost, self.trainer], feed_dict={
                        self.state_input: state_batch,
                        self.Q_gt: Q_gt,
                        self.action: action_batch})
    self.update_target_network()
    return c

  def get_action(self):
    action_pred = self.session.run([self.action_pred], \
            feed_dict={self.state_input: self.cur_state[np.newaxis, :]})[0]
    action = np.zeros(FLAGS.num_actions)
    action_ind = action_pred if random.random() > self.epsilon \
                             else random.randrange(FLAGS.num_actions)
    action[action_ind] = 1

    if self.epsilon > FLAGS.end_e and self.step > FLAGS.observe:
        self.epsilon -= (FLAGS.start_e - FLAGS.end_e)/FLAGS.explore
    return action

  #def proc_state(self, state):
