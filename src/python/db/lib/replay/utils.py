#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 12 May 08:33:24 2011 

"""Some utilities to talk to the replay attack database.
"""

import os, sys

def location():
  """Returns the location of the database"""

  fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'db.sql3')
  return 'sqlite:///%s' % fname

def session(echo=False):
  """Creates a session to the Replay attack database"""
  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  engine = create_engine(location(), echo=echo)
  Session = sessionmaker(bind=engine)
  return Session()

def dbshell():
  """Drops you into a database shell"""
  import subprocess

  arguments = ['sqlite3', location().replace('sqlite:///','')]

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
