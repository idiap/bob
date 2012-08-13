#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 13 Aug 2012 16:19:18 CEST 

"""An abstract class that defines the required behavior of implemented
databases.
"""

import abc

class Database(object):
  """Base type for Bob databases"""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def name(self):
    '''Returns a simple name for this database, w/o funny characters, spaces'''
    return

  @abc.abstractmethod
  def location(self):
    '''Returns the directory that contains the data'''
    return

  @abc.abstractmethod
  def files(self):
    '''Returns a python iterable with all auxiliary files needed.
    
    The values should be take w.r.t. where the python file that declares the
    database is sitting at.
    '''
    return

  @abc.abstractmethod
  def version(self):
    '''Returns the current version number defined in setup.py'''
    return

  @abc.abstractmethod
  def type(self):
    '''Returns the type of auxiliary files you have for this database
    
    If you return 'sqlite', then we append special actions such as 'dbshell'
    on 'bob_dbmanage.py' automatically for you. Otherwise, we don't.

    If you use auxiliary text files, just return 'text'. We may provide
    special services for those types in the future.

    Use the special name 'builtin' if this database is an integral part of Bob.
    '''
    return

  @abc.abstractmethod
  def add_commands(self, parser):
    '''Adds commands to a given (argparse) parser.
    
    This method, effectively, allows you to define special commands that your
    database will be able to perform when called from the common driver like
    for example ``create`` or ``checkfiles``.
    '''
    return
