import os
import numpy as np

class DataLoader(object):
  # trainTestRatio is the train/test switch ratio for learning training procedure
  def __init__(self, directory, trainTestRatio=0.8):
    train_path = os.path.join(directory, 'train')
    files = [f for f in os.listdir(train_path) \
             if os.path.isfile(os.path.join(train_path, f))]
    #print files
    self.n_train_samples = len(files)
    self.train_x = {}
    self.train_y = {}
    for _file in files:
      file_data = np.loadtxt(os.path.join(train_path, _file))
      ind = int(os.path.splitext(os.path.basename(_file))[0])-1
      self.train_x[ind] = file_data[:, 1:]
      n_instances = self.train_x[ind].shape[0]
      self.train_y[ind] = np.zeros([n_instances, 2])
      self.train_y[ind][np.arange(n_instances), file_data[:, 0].astype(int)] = 1
      # print self.train_y[ind]

    test_path = os.path.join(directory, 'test')
    files = [f for f in os.listdir(test_path) \
             if os.path.isfile(os.path.join(test_path, f))]
    self.n_test_samples = len(files)
    self.test_x = {}
    self.test_y= {}
    for _file in files:
      file_data = np.loadtxt(os.path.join(test_path, _file))
      ind = int(os.path.splitext(os.path.basename(_file))[0])-1
      self.test_x[ind] = file_data[:, 1:]
      n_instances = self.test_x[ind].shape[0]
      self.test_y[ind] = np.zeros([n_instances, 2])
      self.test_y[ind][np.arange(n_instances), file_data[:, 0].astype(int)] = 1

    self.n_instances = self.train_x[0].shape[0]
    self.trainTestRatio = trainTestRatio

  # isTrainingOpt is the train/test switch for learning to learn training procedure
  # isTraining is the train/test switch for learning training procedure
  def next_batch(self, sample_ind, batch_size, isTraining=True, isTrainingOpt=True):
    n_training_instances = int(self.n_instances*self.trainTestRatio)
    inds = np.random.randint(n_training_instances, size=batch_size) if isTraining \
            else n_training_instances + np.random.randint(self.n_instances-n_training_instances, \
            size=batch_size)
    x = self.train_x[sample_ind][inds, :] if isTrainingOpt else self.test_x[sample_ind][inds, :]
    y = self.train_y[sample_ind][inds, :] if isTrainingOpt else self.test_y[sample_ind][inds, :]
    return x, y

if __name__ == "__main__":
  dataloader = DataLoader('./')
  for i in xrange(3):
    x, y = dataloader.next_batch(0, 3)
    print x
    print y
