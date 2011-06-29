#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 12 May 08:33:24 2011 

"""Some utilities to talk to torch SQLite databases.
"""

import os, sys

class null(object):
  """A look-alike stream that discards the input"""

  def write(self, s):
    pass
  
  def flush(self):
    pass

def location(dbname):
  """Returns the location of the database. The location of the database, by
  default, is the directory containing the database library. If the environment
  variable $TORCH_DB_DIR is defined, it is used as the base-location of the
  file."""
  
  envvar = 'TORCH_DB_DIR'

  if os.environ.has_key(envvar) and os.environ[envvar].strip():
    dirname = os.path.realpath(os.environ[envvar])
    fname = os.path.join(dirname, dbname + '.sql3')

  else:
    dirname = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dirname, dbname, 'db.sql3')

  return 'sqlite:///%s' % fname

def session(dbname, echo=False):
  """Creates a session to an SQLite database"""

  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  engine = create_engine(location(dbname), echo=echo)
  Session = sessionmaker(bind=engine)
  return Session()

def dbshell(dbname):
  """Drops you into a database shell"""

  import subprocess

  arguments = ['sqlite3', location(dbname).replace('sqlite:///','')]

  try:
    p = subprocess.Popen(arguments)
  except OSError as e:
    # occurs when the file is not executable or not found
    print("Error executing '%s': %s (%d)" % (' '.join(arguments), e.strerror,
        e.errno))
    sys.exit(e.errno)
  
  try:
    p.communicate()
  except KeyboardInterrupt: # the user CTRL-C'ed
    import signal
    os.kill(p.pid, signal.SIGTERM)
    return signal.SIGTERM

  return p.returncode
