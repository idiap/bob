#!/usr/bin/env python
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 23 Jun 20:22:28 2011 CEST 
# vim: set fileencoding=utf-8 :

import os
import sys
import numpy

from ..database import Database as AbstractDatabase

class Database(AbstractDatabase):
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

  def name(): 
    '''Returns a simple name for this database, w/o funny characters, spaces'''
    return 'iris'

  def location():
    '''Returns the directory that contains the data'''
    return os.path.dirname(__file__)

  def files():
    '''Returns a python iterable with all auxiliary files needed.
    
    The values should be take w.r.t. where the python file that declares the
    database is sitting at.
    '''
    
    return ('iris.names', 'iris.data')

  def version():
    '''Returns the current version number from Bob's build'''

    from ...build import version
    return version

  def type():
    '''Returns the type of auxiliary files you have for this database
    
    If you return 'sqlite', then we append special actions such as 'dbshell'
    on 'bob_dbmanage.py' automatically for you. Otherwise, we don't.

    If you use auxiliary text files, just return 'text'. We may provide
    special services for those types in the future.

    Use the special name 'builtin' if this database is an integral part of Bob.
    '''

    return 'builtin'

  def data(self):
    """Loads Fisher's Iris Dataset.
    
    This set is small and simple enough to require an SQL backend. We keep
    the single file it has in text and load it on-the-fly every time this
    method is called.

    We return a dictionary containing the 3 classes of Iris plants
    catalogued in this dataset. Each dictionary entry contains an Arrayset
    of 64-bit floats and 50 entries. Each entry is an Array with 4
    features as described by "names". 
    """
    from ...io import Arrayset
    
    data = os.path.join(self.location(), 'iris.data')

    retval = {
        'setosa': Arrayset(),
        'versicolor': Arrayset(),
        'virginica': Arrayset()
        }

    for line in open(data,'rt'):
      if not line.strip(): continue #skip empty lines

      s = [k.strip() for k in line.split(',') if line.strip()]

      if s[4].find('setosa') != -1:
        retval['setosa'].append(numpy.array([float(k) for k in s[0:4]], 'float64'))

      elif s[4].find('versicolor') != -1:
        retval['versicolor'].append(numpy.array([float(k) for k in s[0:4]], 'float64'))

      elif s[4].find('virginica') != -1:
        retval['virginica'].append(numpy.array([float(k) for k in s[0:4]], 'float64'))

    return retval

  def dump(self, args):

    d = self.data()
    if args.cls: d = {args.cls: d[args.cls]}

    output = sys.stdout
    if args.selftest:
      from ..utils import null
      output = null()

    for k, v in d.items():
      for array in v:
        s = ','.join(['%.1f' % array[i] for i in range(array.shape[0])] + [k])
        output.write('%s\n' % (s,))

  def add_commands(self, parser):

    """A few commands this database can respond to."""
    
    from argparse import RawDescriptionHelpFormatter, SUPPRESS

    # creates a top-level parser for this database
    top_level = parser.add_parser(self.name(),
        formatter_class=RawDescriptionHelpFormatter,
        help="Fisher's Iris plants database", description=__doc__)

    # declare it has subparsers for each of the supported commands
    subparsers = top_level.add_subparsers(title="subcommands")

    # get the "dumplist" action from a submodule
    dump_message = "Dumps the database in comma-separate-value format"
    dump_parser = subparsers.add_parser('dump', help=dump_message)
    dump_parser.add_argument('-c', '--class', dest="cls", default='', help="if given, limits the dump to a particular subset of the data that corresponds to the given class (defaults to '%(default)s')", choices=('setosa', 'versicolor', 'virginica', ''))
    dump_parser.add_argument('--self-test', dest="selftest", default=False,
        action='store_true', help=SUPPRESS)

    dump_parser.set_defaults(func=self.dump)

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
