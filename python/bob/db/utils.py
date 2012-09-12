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
  from sqlite3 import connect

  try:
    connect(':memory:', uri=True)
    return True
  except TypeError:
    return False

class SQLiteConnector(object):
  '''An object that handles the connection to SQLite databases.'''

  @staticmethod
  def filesystem_is_lockable(database):
    """Checks if the filesystem is lockable"""
    from sqlite3 import connect, OperationalError

    old = os.path.exists(database) #memorize if the database was already there
    conn = connect(database)

    retval = True
    try:
      conn.execute('PRAGMA synchronous = OFF')
    except OperationalError:
      retval = False
    finally:
      if not old and os.path.exists(database): os.unlink(database)

    return retval

  SQLITE3_WITH_URIS = sqlite3_accepts_uris()

  def __init__(self, filename, readonly=False, lock=None):
    """Initializes the connector

    Keyword arguments

    filename
      The name of the file containing the SQLite database

    readonly
      Should I try and open the database in read-only mode? URI support 
      is required. If URI's are not supported, a warning is emitted.

    lock
      Use the given lock system. If not specified, use the default. Values that
      can be given corresponds to the locking capabilities of the SQLite
      driver. For UNIX filesystems the default list is:

      unix-dotfile
        uses dot-file locking rather than POSIX advisory locks.

      unix-excl
        obtains and holds an exclusive lock on database files, preventing other
        processes from accessing the database. Also keeps the wal-index in heap
        rather than in shared memory.

      unix-none
        all file locking operations are no-ops.

      unix-namedsem
        uses named semaphores for file locking. VXWorks only.

    """

    opts = {}
    if readonly: opts['mode'] = 'ro'
    if isinstance(lock, (str, unicode)): opts['vfs'] =  lock

    self.lockable = SQLiteConnector.filesystem_is_lockable(filename)

    if opts and not self.SQLITE3_WITH_URIS:

      if not self.lockable:
        import warnings
        warnings.warn('Got a request for an SQLite connection with options, but SQLite connection options are not supported at the installed version of Python (check http://bugs.python.org/issue13773 for a discussion and a patch). Furthermore, the place where the database is sitting ("%s") is on a filesystem that does **not** seem to support locks. I\'m returning a connection and hopping for the best.' % (filename,))

      # Note: the warning will only come if you are in a filesystem that does
      # not support locks.
      opts = {}

    if self.SQLITE3_WITH_URIS:
      ostr = ''
      if opts:
        ostr = '?' + '&'.join([k + '=' + v for (k, v) in opts.iteritems()])
      self.uri = 'file:' + filename + ostr
    else:
      self.uri = filename
    
  def __call__(self):

    from sqlite3 import connect

    if self.SQLITE3_WITH_URIS:
      return connect(self.uri, uri=True)

    return connect(self.uri)

  def create_engine(self, echo=False):
    """Returns an SQLAlchemy engine"""

    from sqlalchemy import create_engine
    return create_engine('sqlite://', creator=self, echo=echo)

  def session(self, echo=False):
    """Returns an SQLAlchemy session"""
  
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=self.create_engine(echo))
    return Session()

def session(dbtype, dbfile, echo=False):
  """Creates a session to an SQLite database"""

  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  url = connection_string(dbtype, dbfile)
  engine = create_engine(url, echo=echo)
  Session = sessionmaker(bind=engine)
  return Session()

def session_try_readonly(dbtype, dbfile, echo=False):
  """Creates a read-only session to an SQLite database. If read-only sessions 
  are not supported by the underlying sqlite3 python DB driver, then a normal
  session is returned. A warning is emitted in case the underlying filesystem
  does not support locking properly.
  
  Raises a RuntimeError if the file does not exist.
  Raises a NotImplementedError if the dbtype is not supported.
  """

  if dbtype != 'sqlite':
    raise NotImplementedError, "Read-only sessions are only currently supported for SQLite databases"

  if not os.path.exists(dbfile):
    raise RuntimeError, "Cannot open **read-only** SQLite session to a file that does not exist (%s)" % dbfile

  connector = SQLiteConnector(dbfile, readonly=True, lock='unix-none')
  return connector.session(echo=echo)

def create_engine_try_nolock(dbtype, dbfile, echo=False):
  """Creates an engine connected to an SQLite database with no locks. If
  engines without locks are not supported by the underlying sqlite3 python DB
  driver, then a normal engine is returned. A warning is emitted if the
  underlying filesystem does not support locking properly in this case.

  Raises a NotImplementedError if the dbtype is not supported.
  """

  if dbtype != 'sqlite':
    raise NotImplementedError, "Unlocked engines are only currently supported for SQLite databases"

  connector = SQLiteConnector(dbfile, lock='unix-none')
  return connector.create_engine(echo=echo)

def session_try_nolock(dbtype, dbfile, echo=False):
  """Creates a session to an SQLite database with no locks. If sessions without
  locks are not supported by the underlying sqlite3 python DB driver, then a
  normal session is returned. A warning is emitted if the underlying filesystem
  does not support locking properly in this case.

  Raises a NotImplementedError if the dbtype is not supported.
  """
  
  if dbtype != 'sqlite':
    raise NotImplementedError, "Unlocked sessions are only currently supported for SQLite databases"

  connector = SQLiteConnector(dbfile, lock='unix-none')
  return connector.session(echo=echo)

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
