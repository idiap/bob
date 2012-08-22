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

def session(dbtype, dbdir, dbfile, echo=False):
  """Creates a session to an SQLite database"""

  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  engine = create_engine(connection_string(dbtype, dbdir, dbfile), echo=echo)
  Session = sessionmaker(bind=engine)
  return Session()

def connection_string(dbtype, dbdir, dbfile):
  """Returns a connection string for supported platforms"""

  if dbtype == 'sqlite': return 'sqlite:///' + os.path.join(dbdir, dbfile)
    
  # otherwise, we just raise
  raise RuntimeError, "Cannot create connection strings for database types different than 'sqlite' and you have type='%s'" % (dbtype,)

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
