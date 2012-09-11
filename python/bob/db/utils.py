#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 12 May 08:33:24 2011 

"""Some utilities shared by many of the databases.
"""

import os
import sys
import errno
import stat
import logging
import sqlite3

class null(object):
  """A look-alike stream that discards the input"""

  def write(self, s):
    """Writes contents of string ``s`` on this stream"""

    pass
  
  def flush(self):
    """Flushes the stream"""

    pass

def sqlite3_accepts_uris():
  """Checks lock-ability for SQLite on the current file system"""

  try:
    sqlite3.connect(':memory:', uri=True)
    return True
  except TypeError:
    return False

class SQLiteConnector(object):
  '''An object that handles the connection to SQLite databases.'''

  SQLITE3_WITH_URIS = sqlite3_accepts_uris()

  def __init__(self, filename, readonly=False, lock=None):

    opts = {}
    if readonly: opts['mode'] = 'ro'
    if isinstance(lock, (str, unicode)): opts['vfs'] =  lock

    if opts and not self.SQLITE3_WITH_URIS:
      import warnings
      warnings.warn('Got a request for a connection string to an SQLite session with options, but SQLite connection options are not supported at the installed version of Python (check http://bugs.python.org/issue13773 for a discussion and a patch). Returning a standard connection string.' % ('.'.join(sqlite_version),))
      opts = {}

    if self.SQLITE3_WITH_URIS:
      ostr = ''
      if opts:
        ostr = '?' + '&'.join([k + '=' + v for (k, v) in opts.iteritems()])
      self.uri = 'file:' + filename + ostr
    else:
      self.uri = filename
    
  def __call__(self):

    if self.SQLITE3_WITH_URIS:
      return sqlite3.connect(self.uri, uri=True)

    return sqlite3.connect(self.uri)

def session(dbtype, dbfile, echo=False):
  """Creates a session to an SQLite database"""

  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  url = connection_string(dbtype, dbfile)
  engine = create_engine(url, echo=echo)
  Session = sessionmaker(bind=engine)
  return Session()

def session_readonly(dbtype, dbfile, echo=False):
  """Creates a session to an SQLite database.
  
  Raises a RuntimeError if the file does not exist.
  """

  if dbtype != 'sqlite':
    raise NotImplemented, "Read-only sessions are only currently supported for SQLite databases"

  if not os.path.exists(dbfile):
    raise RuntimeError, "Cannot open **read-only** SQLite session to a file that does not exist (%s)" % dbfile

  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  engine = create_engine('sqlite://', creator=SQLiteConnector(dbfile, readonly=True, lock='unix-none'))
  Session = sessionmaker(bind=engine)
  return Session()

def connection_string(dbtype, dbfile, opts={}):
  """Returns a connection string for supported platforms
  
  Keyword parameters

  dbtype
    The type of database (only 'sqlite' is supported for the time being)

  dbfile
    The location of the file to be used
  """

  from sqlalchemy.engine.url import URL
  return URL(dbtype, database=dbfile)
    
def makedirs_safe(fulldir, dryrun=False):
  """Creates a directory if it does not exists, with concurrent access support"""
  try:
    if dryrun:
      print "[dry-run] mkdir -p '%s'" % fulldir
    else:
      if not os.path.exists(fulldir): os.makedirs(fulldir)

  except OSError as exc: # Python >2.5
    if exc.errno == errno.EEXIST: pass
    else: raise
