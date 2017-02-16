import numpy as np
import tensorflow as tf
import Qnetwork
import sys
from env import SoftmaxRegression as SR

def play(argv):
  env_config = {
  'n_training_samples' : 55000,
  'learning_rate' : 8e-1,
  'n_epochs' : 50,
  'n_batches' : 550, #env_config['n_training_samples']/env_config['batch_size']
  'batch_size' : 100,
  'display_step' : 110,
  }
  print env_config

  env = SR(env_config)
  agent = Qnetwork.Qnetwork()

  for ep in xrange(Qnetwork.FLAGS.num_episodes):
    env.reset()
    action_1st = np.array([0, 1])
    _, _, state, _, _= env.step(action_1st)
    agent.init_state(state)
    epoch, avg_cost, avg_max_grad, avg_min_grad, a0, terminal = 0, 0, 0, 0, 0, False
    while not terminal:
      action = agent.get_action()
      if np.argmax(action) == 0:
        a0 += 1.0
      c, grads, nxt_state, reward, terminal = env.step(action)

      avg_cost += c / env_config['n_batches']
      grads_val_max = np.asarray([grad.max() for grad in grads]).max()
      grads_val_min = np.asarray([grad.min() for grad in grads]).min()
      avg_max_grad += grads_val_max / env_config['n_batches']
      avg_min_grad += grads_val_min / env_config['n_batches']

      state_str = agent.set_perception(action, reward, nxt_state, terminal)
      batch_ind = env.get_n_step() - 1
      if batch_ind % env_config['display_step'] == 0:
        print '  [%02d/%02d/%03d]'%(ep+1, epoch+1, batch_ind), \
              "cost={:.9f}".format(c), "lr={:.5f}".format(env.print_lr()), \
              "grad=({:.5f},{:.5f})".format(grads_val_max, grads_val_min), \
              "r={:.5f} t={:d}".format(reward, terminal), state_str

      if batch_ind % env_config['n_batches'] == env_config['n_batches'] - 1:
        print ' [%02d/%02d]'%(ep+1, epoch+1), \
              "cost={:.9f}".format(avg_cost), "lr={:.5f}".format(env.print_lr()), \
              "avg_grad=({:.5f},{:.5f})".format(avg_max_grad, avg_min_grad), \
              "action0_rate={:.5f}".format(a0/env_config['n_batches'])
        avg_cost, avg_max_grad, avg_min_grad = 0, 0, 0
        epoch += 1

if __name__ == "__main__":
  tf.app.run(play)
