#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 14 Aug 20:34:00 2012 

"""Interface definition for Bob's database driver
"""

from ..driver import Interface as AbstractInterface

class Interface(AbstractInterface):
  """Bob Manager interface for the Iris Flower Database"""

  def name(self): 
    '''Returns a simple name for this database, w/o funny characters, spaces'''
    return 'iris'

  def files(self):
    '''Returns a python iterable with all auxiliary files needed.
    
    The values should be take w.r.t. where the python file that declares the
    database is sitting at.
    '''
    
    from pkg_resources import resource_filename
    raw_files = ('iris.data', 'iris.names')
    return [resource_filename(__name__, k) for k in raw_files]

  def version(self):
    '''Returns the current version number from Bob's build'''

    import pkg_resources  # part of setuptools
    version = pkg_resources.require('bob')[0].version
    return version + ' (built-in)'

  def type(self):
    '''Returns the type of auxiliary files you have for this database
    
    If you return 'sqlite', then we append special actions such as 'dbshell'
    on 'bob_dbmanage.py' automatically for you. Otherwise, we don't.

    If you use auxiliary text files, just return 'text'. We may provide
    special services for those types in the future.

    Use the special name 'builtin' if this database is an integral part of Bob.
    '''

    return 'builtin'

  def add_commands(self, parser):

    """A few commands this database can respond to."""

    from argparse import SUPPRESS
    from . import __doc__ as docs
    
    subparsers = self.setup_parser(parser, "Fisher's Iris Flower dataset", docs)

    # get the "dumplist" action from a submodule
    dump_message = "Dumps the database in comma-separate-value format"
    dump_parser = subparsers.add_parser('dump', help=dump_message)
    dump_parser.add_argument('-c', '--class', dest="cls", default='', help="if given, limits the dump to a particular subset of the data that corresponds to the given class (defaults to '%(default)s')", choices=('setosa', 'versicolor', 'virginica', ''))
    dump_parser.add_argument('--self-test', dest="selftest", default=False,
        action='store_true', help=SUPPRESS)

    from . import __dump__
    dump_parser.set_defaults(func=__dump__)
