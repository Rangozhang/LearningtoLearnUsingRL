import numpy as np
import tensorflow as tf
import qnet
import sys, time
from env import SoftmaxRegression as SR

def play(argv):
  env_config = {
  'n_training_samples' : 55000,
  'learning_rate' : 4e-3,
  'n_epochs' : 50,
  'n_batches' : 550, #env_config['n_training_samples']/env_config['batch_size']
  'batch_size' : 100,
  'display_step' : 110,
  }
  print env_config

  env = SR(env_config)
  agent = qnet.qnet()

  datetime = int(time.time())
  loss_holder = tf.placeholder(tf.float32)
  lr_holder = tf.placeholder(tf.float32)
  avg_lr_holder = tf.placeholder(tf.float32)
  loss_summary = tf.summary.scalar('loss', loss_holder)
  avg_lr_summary = tf.summary.scalar('avg_lr', avg_lr_holder)
  lr_summary = tf.summary.scalar('lr', lr_holder)
  merged_epoch_summary = tf.summary.merge([loss_summary, avg_lr_summary])
  merged_step_summary = tf.summary.merge([lr_summary])

  for ep in xrange(qnet.FLAGS.num_episodes):
    env.reset()
    action_1st = np.array([0, 1])
    _, _, state, _, _= env.step(action_1st)
    agent.init_state(state)

    writer = tf.summary.FileWriter("./res/train_fig/{:10d}/ep{:d}".format(datetime, ep))

    epoch, avg_lr, avg_cost, avg_max_grad, avg_min_grad, a0, a1, terminal = 0, 0, 0, 0, 0, 0, 0, False
    avg_r0, avg_r1 = 0, 0
    while not terminal:
      action = agent.get_action()
      c, grads, nxt_state, reward, terminal = env.step(action)
      if np.argmax(action) == 0:
        a0 += 1.0
        avg_r0 += reward
      elif np.argmax(action) == 1:
        a1 += 1.0
        avg_r1 += reward

      avg_lr += env.print_lr() / env_config['n_batches']
      avg_cost += c / env_config['n_batches']
      grads_val_max = np.asarray([grad.max() for grad in grads]).max()
      grads_val_min = np.asarray([grad.min() for grad in grads]).min()
      avg_max_grad += grads_val_max / env_config['n_batches']
      avg_min_grad += grads_val_min / env_config['n_batches']

      state_str = agent.set_perception(action, reward, nxt_state, terminal)
      batch_ind = env.get_n_step() - 1

      if batch_ind % env_config['display_step'] == 0:
        print '  [%02d/%02d/%03d]'%(ep+1, epoch+1, batch_ind), \
              "cost={:.9f}".format(c), "lr={:.9f}".format(env.print_lr()), \
              "grad=({:.5f},{:.5f})".format(grads_val_max, grads_val_min), \
              "r={:.5f} t={:d}".format(reward, terminal), state_str, \
              "a={:d}".format(np.argmax(action))

      if batch_ind % env_config['n_batches'] == env_config['n_batches'] - 1:
        record_epoch_summary = env.session.run([merged_epoch_summary], feed_dict={loss_holder: avg_cost,
                                                                                  avg_lr_holder: avg_lr})[0]
        writer.add_summary(record_epoch_summary, epoch)
        print ' [%02d/%02d]'%(ep+1, epoch+1), \
              "avg_cost={:.9f}".format(avg_cost), "avg_lr={:.9f}".format(avg_lr), \
              "avg_grad=({:.5f},{:.5f})".format(avg_max_grad, avg_min_grad), \
              "action_rate=[{:.2f}, {:.2f}]".format(a0/env_config['n_batches'], a1/env_config['n_batches']), \
              "avg_reward=[{:.5f}, {:.5f}]".format(avg_r0/a0, avg_r1/a1)
        avg_lr, avg_cost, avg_max_grad, avg_min_grad, a0, a1 = 0, 0, 0, 0, 0, 0
        epoch += 1
        avg_r0, avg_r1 = 0, 0

      record_step_summary = env.session.run([merged_step_summary], feed_dict={lr_holder: env.print_lr()})[0]
      writer.add_summary(record_step_summary, batch_ind)

if __name__ == "__main__":
  tf.app.run(play)
