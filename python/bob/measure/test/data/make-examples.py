#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Apr 20 17:32:54 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Separate the Iris dataset to make up toy examples for us.

For separability information, please consult: http://en.wikipedia.org/wiki/File:Anderson%27s_Iris_data_set.png
"""

import bob

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
  bob.io.save(data['setosa'], 'linsep-negatives.hdf5')
  bob.io.save(data['virginica'],'linsep-positives.hdf5')

def example2():
  """In the second example we will get a non-linearly separable set of scores:

  Variable: Sepal length
  Iris setosa: noise
  Iris versicolor: signal

  Separation threshold is about 5 (min. HTER).
  """
  data = loaddata('iris.data', 'sepal.length')
  bob.io.save(data['setosa'], 'nonsep-negatives.hdf5')
  bob.io.save(data['versicolor'], 'nonsep-positives.hdf5')

def main():
  """Generates data for all examples."""
  example1()
  example2()

if __name__ == '__main__':
  main()
