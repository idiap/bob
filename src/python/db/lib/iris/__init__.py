#!/usr/bin/env python
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 23 Jun 20:22:28 2011 CEST 
# vim: set fileencoding=utf-8 :

"""
The Iris flower data set or Fisher's Iris data set is a multivariate data
set introduced by Sir Ronald Aylmer Fisher (1936) as an example of
discriminant analysis. It is sometimes called Anderson's Iris data set
because Edgar Anderson collected the data to quantify the geographic
variation of Iris flowers in the Gaspé Peninsula.  

For more information: http://en.wikipedia.org/wiki/Iris_flower_data_set
"""

import os
from ...io import Arrayset
from ...core.array import array

names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
"""Names of the features for each entry in the dataset."""

stats = {
    'Sepal Length': [4.3, 7.9, 5.84, 0.83, 0.7826],
    'Sepal Width': [2.0, 4.4, 3.05, 0.43, -0.4194],
    'Petal Length': [1.0, 6.9, 3.76, 1.76, 0.9490], #high correlation
    'Petal Width': [0.1, 2.5, 1.20, 0.76, 0.9565], #high correlation
    }
"""These are basic statistics for each of the features in the whole dataset."""

stat_names = ['Minimum', 'Maximum', 'Mean', 'Std.Dev.', 'Correlation']
"""These are the statistics available in each column of the stats variable."""

def data():
  """Loads Fisher's Iris Dataset.
  
  This set is small and simple enough to require an SQL backend. We keep
  the single file it has in text and load it on-the-fly every time this
  method is called.

  We return a dictionary containing the 3 classes of Iris plants
  catalogued in this dataset. Each dictionary entry contains an Arrayset
  of 64-bit floats and 50 entries. Each entry is an Array with 4
  features as described by "names". 
  """
  
  data = os.path.join(os.path.dirname(__file__), 'iris.data')

  retval = {
      'setosa': Arrayset(),
      'versicolor': Arrayset(),
      'virginica': Arrayset()
      }

  for line in open(data,'rt'):
    if not line.strip(): continue #skip empty lines

    s = [k.strip() for k in line.split(',') if line.strip()]

    if s[4].find('setosa') != -1:
      retval['setosa'].append(array([float(k) for k in s[0:4]], 'float64'))

    elif s[4].find('versicolor') != -1:
      retval['versicolor'].append(array([float(k) for k in s[0:4]], 'float64'))

    elif s[4].find('virginica') != -1:
      retval['virginica'].append(array([float(k) for k in s[0:4]], 'float64'))

  return retval
