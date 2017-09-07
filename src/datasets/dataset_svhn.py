"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

from __future__ import print_function
import dill
import scipy.io
import os
import numpy as np
import torch.utils.data as data
import torch
import urllib
import time

class Cache:
  def __init__(self, name, filepath):
    print ('Initializing cache')
    self.name = name
    self.filepath = filepath
    self.x = None

  def load(self):
    if self.x is not None:
      print('Cache %s is already loaded into memory' % self.name)
    elif not os.path.exists(self.filepath):
      return None
    else:
      print('Loading cached %s into memory... ' % self.name, end='')
      start = time.time()
      with open(self.filepath, 'rb') as f:
        self.x = cPickle.load(f)
      print ('in %f seconds' % (time.time()-start))
    return self.x
    
  def save(self, x):
    print('Caching %s onto disk... ' % self.name, end='')
    start = time.time()
    with open(self.filepath, 'wb') as f:
      cPickle.dump(x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print ('in %f seconds' % (time.time()-start))
    self.x = x


class dataset_svhn_extra(data.Dataset):
# train 73,257, extra 531,131, test, 26,032
  def __init__(self, specs):
    self.url = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
    self.filename = 'extra_32x32.mat'
    self.root = specs['root']
    full_filepath = os.path.join(self.root, self.filename)
    self._download(full_filepath, self.url)
    data_set = self._load_samples(full_filepath)
    self.data = data_set[0]
    self.labels = data_set[1]
    self.num = self.data.shape[0]

  def __getitem__(self, index):
    #img, label = self.data[index, ::], self.labels[index]
    img = self.data[index, ::]
    if self.scale > 0:
      img = imresize(img, size=float(self.scale), interp='bilinear')
    img = torch.FloatTensor(img)
    #label = torch.LongTensor([np.int64(label)])
    return img #, label

  def __len__(self):
    return self.num

  def _download(self, filename, url):
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
      os.mkdir(dirname)
    if os.path.isfile(filename):
      print("%s exists" % filename)
    else:
      start = time.time()
      print("[Download %s to %s]" % (url, filename))
      urllib.urlretrieve(url, filename)
      print("[Downloaded in %f.]" % (time.time() - start))

  def _load_samples(self, filename):
    start = time.time()
    print("[Loading samples %s.]" % os.path.basename(filename))
    mat = scipy.io.loadmat(filename)
    y = mat['y']
    item_index = np.where(y == 10)
    y[item_index] = 0
    x = mat['X']
    train_data = [2*np.float32(np.transpose(x, [3, 2, 0, 1]) / 255.0)-1,
        np.squeeze(y)]
    print("[Loaded in %f.]" % (time.time() - start))
    return train_data

class dataset_svhn_test(dataset_svhn_extra):
# train 73,257, extra 531,131, test, 26,032
  def __init__(self, specs):
    self.url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    self.filename = 'test_32x32.mat'
    self.root = specs['root']
    full_filepath = os.path.join(self.root, self.filename)
    self._download(full_filepath, self.url)
    data_set = self._load_samples(full_filepath)
    self.data = data_set[0]
    self.labels = data_set[1]
    self.num = self.data.shape[0]


class dataset_svhn32x32_0_extra(dataset_svhn_extra):
# train 73,257, extra 531,131, test, 26,032
  cache = None
  def __init__(self, specs):
    self.scale = specs['scale']
    self.root = specs['root']
    if self.cache is None:
      cachepath = os.path.join(self.root, 'svhn32x32_0.pkl')
      self.cache = Cache(name='svhn_0_extra', filepath=cachepath)
    self.data = self.cache.load()
    if self.data is None:
      # it's not in memory or on disk, need to prepare and cache.
      self.url = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
      self.filename = 'extra_32x32.mat'
      full_filepath = os.path.join(self.root, self.filename)
      self._download(full_filepath, self.url)
      data_set = self._load_samples(full_filepath)
      self.data = data_set[0]
      self.labels = data_set[1]
      num_before = self.data.shape[0]
      # keep only those with labels == 0
      mask = self.labels == 0
      self.data = self.data[mask]
      self.labels = self.labels[mask]
      num_after = self.data.shape[0]
      print ('%d out of %d equal to 0' % (num_after, num_before))
      self.cache.save(self.data)
    self.num = self.data.shape[0]

class dataset_svhn32x32_1_extra(dataset_svhn_extra):
# train 73,257, extra 531,131, test, 26,032
  cache = None
  def __init__(self, specs):
    self.scale = specs['scale']
    self.root = specs['root']
    if self.cache is None:
      cachepath = os.path.join(self.root, 'svhn32x32_0.pkl')
      self.cache = Cache(name='svhn_0_train', filepath=cachepath)
    self.data = self.cache.load()
    if self.data is None:
      # it's not in memory or on disk, need to prepare and cache.
      self.url = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
      self.filename = 'extra_32x32.mat'
      full_filepath = os.path.join(self.root, self.filename)
      self._download(full_filepath, self.url)
      data_set = self._load_samples(full_filepath)
      self.data = data_set[0]
      self.labels = data_set[1]
      num_before = self.data.shape[0]
      # keep only those with labels == 1
      mask = self.labels == 1
      self.data = self.data[mask]
      self.labels = self.labels[mask]
      num_after = self.data.shape[0]
      print ('%d out of %d equal to 1' % (num_after, num_before))
      self.cache.save(self.data)
    self.num = self.data.shape[0]

