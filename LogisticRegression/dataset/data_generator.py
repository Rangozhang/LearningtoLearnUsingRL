import os
#import random
import numpy as np

# config
n_dim = 10
n_train_samples = 90
n_test_samples = 100
n_instances_per_sample = 100

def sample(n_instances):
  data_list = []

  mean = np.random.uniform(-2, 2, size=n_dim)
  cov = np.diagflat(np.random.uniform(0.8, 1, size=n_dim))
  #print mean, cov

  data_list = np.random.multivariate_normal(mean, cov, n_instances)
  #print data_list.shape
  return data_list

if __name__ == "__main__":
  if not os.path.exists('train'):
    os.makedirs('train')

  if not os.path.exists('test'):
    os.makedirs('test')

  n_instances_per_gauss = n_instances_per_sample/2

  for i in xrange(n_train_samples):
    label_list = np.zeros(n_instances_per_sample)
    label_list[n_instances_per_gauss:] = 1

    data_list = sample(n_instances_per_gauss)
    data_list = np.append(data_list, sample(n_instances_per_gauss), axis=0)

    tmp = np.append(np.expand_dims(label_list, axis=1), data_list, axis=1)
    np.random.shuffle(tmp)
    np.savetxt(os.path.join('train','{}.txt'.format(i+1)), tmp)

  for i in xrange(n_train_samples):
    label_list = np.zeros(n_instances_per_sample)
    label_list[n_instances_per_gauss:] = 1

    data_list = sample(n_instances_per_gauss)
    data_list = np.append(data_list, sample(n_instances_per_gauss), axis=0)

    tmp = np.append(np.expand_dims(label_list, axis=1), data_list, axis=1)
    np.random.shuffle(tmp)
    np.savetxt(os.path.join('test','{}.txt'.format(i+1)), tmp)
