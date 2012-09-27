#!/usr/bin/env python
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 23 Jun 20:22:28 2011 CEST 
# vim: set fileencoding=utf-8 :

"""
The Iris flower data set or Fisher's Iris data set is a multivariate data
set introduced by Sir Ronald Aylmer Fisher (1936) as an example of
discriminant analysis. It is sometimes called Anderson's Iris data set
because Edgar Anderson collected the data to quantify the geographic
variation of Iris flowers in the Gaspe Peninsula.  

For more information: http://en.wikipedia.org/wiki/Iris_flower_data_set

References:

  1. Fisher,R.A. "The use of multiple measurements in taxonomic problems",
  Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
  Mathematical Statistics" (John Wiley, NY, 1950).

  2. Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
  (Q327.D83) John Wiley & Sons. ISBN 0-471-22361-1. See page 218. 

  3. Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
  Structure and Classification Rule for Recognition in Partially Exposed
  Environments". IEEE Transactions on Pattern Analysis and Machine
  Intelligence, Vol. PAMI-2, No. 1, 67-71. 

  4. Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule". IEEE
  Transactions on Information Theory, May 1972, 431-433. 
"""

import os
import sys
import numpy
from . import driver #driver interface

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
  """Loads from (text) file and returns Fisher's Iris Dataset.
  
  This set is small and simple enough to require an SQL backend. We keep the
  single file it has in text and load it on-the-fly every time this method is
  called.

  We return a dictionary containing the 3 classes of Iris plants catalogued in
  this dataset. Each dictionary entry contains an 2D :py:class:`numpy.ndarray`
  of 64-bit floats and 50 entries. Each entry is an Array with 4 features as
  described by "names".
  """
  from .driver import Interface
  import csv

  data = Interface().files()[0]

  retval = {}
  with open(data, 'rb') as csvfile:
    for row in csv.reader(csvfile):
      name = row[4][5:].lower()
      retval.setdefault(name, []).append([float(k) for k in row[:4]])

  # Convert to a float64 2D numpy.ndarray
  for key, value in retval.iteritems():
    retval[key] = numpy.array(value, dtype='float64')

  return retval

def __dump__(args):
  """Dumps the database to stdout.

  Keyword arguments:

  args
    A argparse.Arguments object with options set. We use two of the options:
    ``cls`` for the class to be dumped (if None, then dump all data) and
    ``selftest``, which runs the internal test.
  """

  d = data()
  if args.cls: d = {args.cls: d[args.cls]}

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  for k, v in d.items():
    for array in v:
      s = ','.join(['%.1f' % array[i] for i in range(array.shape[0])] + [k])
      output.write('%s\n' % (s,))

  return 0

__all__ = ['names', 'stats', 'stat_names', 'data']
