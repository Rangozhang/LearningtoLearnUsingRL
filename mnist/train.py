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
  'display_step' : np.Inf,
  'action_freq' : 110,
  }
  print env_config

  env = SR(env_config)
  agent = qnet.qnet()

  datetime = int(time.time())
  print datetime

  loss_holder = tf.placeholder(tf.float32)
  cost_holder = tf.placeholder(tf.float32)
  avg_lr_holder = tf.placeholder(tf.float32)
  q_holder = tf.placeholder(tf.float32, [None, 2])
  q_t_holder = tf.placeholder(tf.float32, [None, 2])
  # lr_holder = tf.placeholder(tf.float32)
  # r_holder0 = tf.placeholder(tf.float32, [None])
  # r_holder1 = tf.placeholder(tf.float32, [None])

  cost_summary = tf.summary.scalar('cost', cost_holder)
  loss_summary = tf.summary.scalar('loss', loss_holder)
  avg_lr_summary = tf.summary.scalar('avg_lr', avg_lr_holder)
  q_summary = tf.summary.histogram('q', q_holder)
  q_t_summary = tf.summary.histogram('q', q_t_holder)
  # lr_summary = tf.summary.scalar('lr', lr_holder)
  # r_summary0 = tf.summary.histogram('reward_distribution0', r_holder0)
  # r_summary1 = tf.summary.histogram('reward_distribution1', r_holder1)

  merged_epoch_summary = tf.summary.merge([cost_summary, avg_lr_summary]) # , r_summary0, r_summary1
  merged_summary = tf.summary.merge([loss_summary, q_summary, q_t_summary])
  # merged_step_summary = tf.summary.merge([lr_summary])

  writer = tf.summary.FileWriter("./res/train_fig/envlr{:.5f}_{:10d}/".format(env_config['learning_rate'], datetime))

  for ep in xrange(qnet.FLAGS.num_episodes):
    print "[{}]".format(ep+1)
    env.reset()
    action_1st = np.array([0, 1])
    _, _, state, _, _= env.step(action_1st)
    agent.init_state(state)
    env.reset()

    ep_writer = tf.summary.FileWriter("./res/train_fig/envlr{:.5f}_{:10d}/ep{:d}".format(env_config['learning_rate'], datetime, ep+1))

    avg_r0, avg_r1, epoch, avg_loss, avg_lr, avg_cost, avg_max_grad, avg_min_grad, a0, a1, terminal \
                                                                            = 0, 0, 0, 0, 0, 0, 0, 0, 1e-10, 1e-10, False
    # r_list0, r_list1 = [], []
    while not terminal:
      # step 1
      action = agent.get_action()

      # step 2
      c, grads, nxt_state, reward, terminal = env.step(action)

      # step 3
      state_str, loss, q_pred, q_pred_t = agent.set_perception(action, reward, nxt_state, terminal)

      if np.argmax(action) == 0:
        a0 += 1.0
        avg_r0 += reward
        # r_list0.append(reward)
      elif np.argmax(action) == 1:
        a1 += 1.0
        avg_r1 += reward
        # r_list1.append(reward)

      avg_lr += env.print_lr() / (env_config['n_batches']/env_config['action_freq'])
      avg_cost += c / env_config['n_batches']
      avg_loss += loss / (env_config['n_batches']/env_config['action_freq'])

      # grads_val_max = np.asarray([grad.max() for grad in grads]).max()
      # grads_val_min = np.asarray([grad.min() for grad in grads]).min()
      # avg_max_grad += grads_val_max / (env_config['n_batches']/env_config['action_freq'])
      # avg_min_grad += grads_val_min / (env_config['n_batches']/env_config['action_freq'])

      batch_ind = env.get_n_step() - 1

      if batch_ind % env_config['display_step'] == 0 and 0 != batch_ind:
        print '  [%02d/%02d/%03d]'%(ep+1, epoch+1, batch_ind), \
              "cost={:.9f}".format(c/env_config['action_freq']), "lr={:.9f}".format(env.print_lr()), \
              "r={:.5f} t={:d}".format(reward, terminal), state_str, \
              "a={:d}".format(np.argmax(action)), \
              "loss={:.5f}".format(loss), \
              "test_accu={:.5f}".format(env.test())
              # "grad=({:.3f},{:.3f})".format(grads_val_max, grads_val_min), \

      if batch_ind % (env_config['n_batches']/env_config['action_freq']) == \
              (env_config['n_batches']/env_config['action_freq']) - 1:
        record_epoch_summary, record_summary = env.session.run([merged_epoch_summary, merged_summary], feed_dict={loss_holder: avg_loss,
                                                                                                       cost_holder: avg_cost,
                                                                                                       avg_lr_holder: avg_lr,
                                                                                                       q_holder: q_pred,
                                                                                                       q_t_holder: q_pred_t})
                                                                                                       # r_holder0: r_list0,
                                                                                                       # r_holder1: r_list1

        ep_writer.add_summary(record_epoch_summary, epoch)
        writer.add_summary(record_summary, ep*(env_config['n_epochs'])+epoch)
        print ' [%02d/%02d]'%(ep+1, epoch+1), \
              "cost={:.9f}".format(avg_cost), "lr={:.9f}".format(avg_lr), \
              state_str, \
              "a/=[{:.2f}, {:.2f}]".format(a0/(env_config['n_batches']/env_config['action_freq']), \
                                                    a1/(env_config['n_batches']/env_config['action_freq'])), \
              "r/=[{:.5f}, {:.5f}]".format(avg_r0/a0, avg_r1/a1), \
              "loss={:.5f}".format(avg_loss)
              # "avg_grad=({:.5f},{:.5f})".format(avg_max_grad, avg_min_grad), \
        avg_r0, avg_r1, avg_loss, avg_lr, avg_cost, avg_max_grad, avg_min_grad, a0, a1 = 0, 0, 0, 0, 0, 0, 0, 1e-10, 1e-10
        epoch += 1
        # r_list0, r_list1 = [], []

      # record_step_summary = env.session.run([merged_step_summary], feed_dict={lr_holder: env.print_lr()})[0]
      # ep_writer.add_summary(record_step_summary, batch_ind)

if __name__ == "__main__":
  tf.app.run(play)
