#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 20 Apr 2011 15:01:03 CEST 

"""Separate the Iris dataset to make up toy examples for us.

For separability information, please consult: http://en.wikipedia.org/wiki/File:Anderson%27s_Iris_data_set.png
"""

import torch

iris_columns = {
    'sepal.length': 0,
    'sepal.width' : 1,
    'petal.length': 2,
    'petal.width' : 3,
    'class'       : 4,
    }

def loaddata(filename, column):
  """Loads the Iris dataset, returns a list with the values"""
  retval = {'setosa': [], 'versicolor': [], 'virginica': []}
  for l in open(filename, 'rt'):
    s = [k.strip() for k in l.split(',')]
    if s[iris_columns['class']] == 'Iris-setosa':
      retval['setosa'].append(float(s[iris_columns[column]]))
    elif s[iris_columns['class']] == 'Iris-versicolor':
      retval['versicolor'].append(float(s[iris_columns[column]]))
    elif s[iris_columns['class']] == 'Iris-virginica':
      retval['virginica'].append(float(s[iris_columns[column]]))
    else:
      raise RuntimeError, 'Unknown data class: %s' % l
  return retval

def example1():
  """In the first example we will get a linearly separable set of scores:
    
  Variable: Petal length
  Iris setosa: noise
  Iris virginica: signal

  Separation threshold is about 3.
  """
  data = loaddata('iris.data', 'petal.length')
  torch.io.Array(torch.core.array.array(data['setosa'])).save('linsep-negatives.hdf5')
  torch.io.Array(torch.core.array.array(data['virginica'])).save('linsep-positives.hdf5')

def example2():
  """In the second example we will get a non-linearly separable set of scores:

  Variable: Sepal length
  Iris setosa: noise
  Iris versicolor: signal

  Separation threshold is about 5 (min. HTER).
  """
  data = loaddata('iris.data', 'sepal.length')
  torch.io.Array(torch.core.array.array(data['setosa'])).save('nonsep-negatives.hdf5')
  torch.io.Array(torch.core.array.array(data['versicolor'])).save('nonsep-positives.hdf5')

def main():
  """Generates data for all examples."""
  example1()
  example2()

if __name__ == '__main__':
  main()
