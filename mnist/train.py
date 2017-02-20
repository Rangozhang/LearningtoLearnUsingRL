import numpy as np
import tensorflow as tf
import qnet
import sys, time
from env import SoftmaxRegression as SR

def play(argv):
  env_config = {
  'n_training_samples' : 55000,
  'learning_rate' : 4e-3,
  'n_epochs' : 30,
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
  r_holder0 = tf.placeholder(tf.float32, [None])
  r_holder1 = tf.placeholder(tf.float32, [None])

  loss_summary = tf.summary.scalar('loss', loss_holder)
  avg_lr_summary = tf.summary.scalar('avg_lr', avg_lr_holder)
  lr_summary = tf.summary.scalar('lr', lr_holder)
  r_summary0 = tf.summary.histogram('reward_distribution0', r_holder0)
  r_summary1 = tf.summary.histogram('reward_distribution1', r_holder1)

  merged_epoch_summary = tf.summary.merge([loss_summary, avg_lr_summary, r_summary0, r_summary1])
  merged_step_summary = tf.summary.merge([lr_summary])

  for ep in xrange(qnet.FLAGS.num_episodes):
    env.reset()
    action_1st = np.array([0, 1])
    _, _, state, _, _= env.step(action_1st)
    agent.init_state(state)

    writer = tf.summary.FileWriter("./res/train_fig/envlr{:.5f}_{:10d}/ep{:d}".format(env_config['learning_rate'], datetime, ep))

    avg_r0, avg_r1, epoch, avg_loss, avg_lr, avg_cost, avg_max_grad, avg_min_grad, a0, a1, terminal \
                                                                                            = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
    r_list0, r_list1 = [], []
    while not terminal:
      # step 1
      action = agent.get_action()

      # step 2
      c, grads, nxt_state, reward, terminal = env.step(action)

      # step 3
      state_str, loss = agent.set_perception(action, reward, nxt_state, terminal)

      if np.argmax(action) == 0:
        a0 += 1.0
        avg_r0 += reward
        r_list0.append(reward)
      elif np.argmax(action) == 1:
        a1 += 1.0
        avg_r1 += reward
        r_list1.append(reward)

      avg_lr += env.print_lr() / env_config['n_batches']
      avg_cost += c / env_config['n_batches']
      grads_val_max = np.asarray([grad.max() for grad in grads]).max()
      grads_val_min = np.asarray([grad.min() for grad in grads]).min()
      avg_max_grad += grads_val_max / env_config['n_batches']
      avg_min_grad += grads_val_min / env_config['n_batches']

      avg_loss += loss / env_config['n_batches']
      batch_ind = env.get_n_step() - 1

      if batch_ind % env_config['display_step'] == 0:
        print '  [%02d/%02d/%03d]'%(ep+1, epoch+1, batch_ind), \
              "cost={:.9f}".format(c), "lr={:.9f}".format(env.print_lr()), \
              "grad=({:.5f},{:.5f})".format(grads_val_max, grads_val_min), \
              "r={:.5f} t={:d}".format(reward, terminal), state_str, \
              "a={:d}".format(np.argmax(action)), \
              "loss={:.5f}".format(loss)

      if batch_ind % env_config['n_batches'] == env_config['n_batches'] - 1:
        record_epoch_summary = env.session.run([merged_epoch_summary], feed_dict={loss_holder: avg_cost,
                                                                                  avg_lr_holder: avg_lr,
                                                                                  r_holder0: r_list0,
                                                                                  r_holder1: r_list1})[0]
        writer.add_summary(record_epoch_summary, epoch)
        print ' [%02d/%02d]'%(ep+1, epoch+1), \
              "avg_cost={:.9f}".format(avg_cost), "avg_lr={:.9f}".format(avg_lr), \
              "avg_grad=({:.5f},{:.5f})".format(avg_max_grad, avg_min_grad), \
              "action_rate=[{:.2f}, {:.2f}]".format(a0/env_config['n_batches'], a1/env_config['n_batches']), \
              "avg_reward=[{:.5f}, {:.5f}]".format(avg_r0/a0, avg_r1/a1), \
              "avg_loss={:.5f}".format(avg_loss)
        avg_r0, avg_r1, avg_loss, avg_lr, avg_cost, avg_max_grad, avg_min_grad, a0, a1 = 0, 0, 0, 0, 0, 0, 0, 0, 0
        epoch += 1
        r_list0, r_list1 = [], []

      record_step_summary = env.session.run([merged_step_summary], feed_dict={lr_holder: env.print_lr()})[0]
      writer.add_summary(record_step_summary, batch_ind)

if __name__ == "__main__":
  tf.app.run(play)
