#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri 20 May 17:00:50 2011 

"""This script creates the BANCA database in a single pass.
"""

import os

from . import dbname
from ..utils import session, location
from .models import *

def add_files(session, imagedir):
  """Add files (and clients) to the BANCA database."""
 
  def add_file(session, filename, client_dict):
    """Parse a single filename and add it to the list.
       Also add a client entry if not already in the database."""

    v = os.path.splitext(os.path.basename(filename))[0].split('_')
    if not (v[0] in client_dict):
      session.add(Client(int(v[0]), v[1], v[2], v[5]))
      client_dict[v[0]] = True
    session_id = int(v[3].split('s')[1])
    session.add(File(int(v[0]), os.path.basename(filename).split('.')[0], v[4], v[6], session_id))
  
  file_list = os.listdir(imagedir)
  client_dict = {} 
  for filename in file_list:
    add_file(session, os.path.join(imagedir, filename), client_dict)


def add_sessions(session):
  """Adds relations between sessions and scenarios"""

  for i in range(1,5):
    session.add(Session(i,'controlled'))
  for i in range(5,9):
    session.add(Session(i,'degraded'))
  for i in range(9,13):
    session.add(Session(i,'adverse'))

def add_protocols(session):
  """Adds protocols"""
  # Protocol Mc
  session.add(Protocol(1, 'Mc', 'enrol'))
  session.add(Protocol(1, 'Mc', 'probeImpostor'))
  session.add(Protocol(2, 'Mc', 'probe'))
  session.add(Protocol(3, 'Mc', 'probe'))
  session.add(Protocol(4, 'Mc', 'probe'))
  
  # Protocol Md
  session.add(Protocol(5, 'Md', 'enrol'))
  session.add(Protocol(5, 'Md', 'probeImpostor'))
  session.add(Protocol(6, 'Md', 'probe'))
  session.add(Protocol(7, 'Md', 'probe'))
  session.add(Protocol(8, 'Md', 'probe'))
  
  # Protocol Ma
  session.add(Protocol(9, 'Ma', 'enrol'))
  session.add(Protocol(9, 'Ma', 'probeImpostor'))
  session.add(Protocol(10, 'Ma', 'probe'))
  session.add(Protocol(11, 'Ma', 'probe'))
  session.add(Protocol(12, 'Ma', 'probe'))
  
  # Protocol Ud
  session.add(Protocol(1, 'Ud', 'enrol'))
  session.add(Protocol(5, 'Ud', 'probeImpostor'))
  session.add(Protocol(6, 'Ud', 'probe'))
  session.add(Protocol(7, 'Ud', 'probe'))
  session.add(Protocol(8, 'Ud', 'probe'))
  
  # Protocol Ma
  session.add(Protocol(1, 'Ua', 'enrol'))
  session.add(Protocol(9, 'Ua', 'probeImpostor'))
  session.add(Protocol(10, 'Ua', 'probe'))
  session.add(Protocol(11, 'Ua', 'probe'))
  session.add(Protocol(12, 'Ua', 'probe'))
  
  # Protocol P
  session.add(Protocol(1, 'P', 'enrol'))
  session.add(Protocol(1, 'P', 'probeImpostor'))
  session.add(Protocol(2, 'P', 'probe'))
  session.add(Protocol(3, 'P', 'probe'))
  session.add(Protocol(4, 'P', 'probe'))
  session.add(Protocol(5, 'P', 'probeImpostor'))
  session.add(Protocol(6, 'P', 'probe'))
  session.add(Protocol(7, 'P', 'probe'))
  session.add(Protocol(8, 'P', 'probe'))
  session.add(Protocol(9, 'P', 'probeImpostor'))
  session.add(Protocol(10, 'P', 'probe'))
  session.add(Protocol(11, 'P', 'probe'))
  session.add(Protocol(12, 'P', 'probe'))
  
  # Protocol G
  session.add(Protocol(1, 'G', 'enrol'))
  session.add(Protocol(5, 'G', 'enrol'))
  session.add(Protocol(9, 'G', 'enrol'))
  session.add(Protocol(1, 'G', 'probeImpostor'))
  session.add(Protocol(2, 'G', 'probe'))
  session.add(Protocol(3, 'G', 'probe'))
  session.add(Protocol(4, 'G', 'probe'))
  session.add(Protocol(5, 'G', 'probeImpostor'))
  session.add(Protocol(6, 'G', 'probe'))
  session.add(Protocol(7, 'G', 'probe'))
  session.add(Protocol(8, 'G', 'probe'))
  session.add(Protocol(9, 'G', 'probeImpostor'))
  session.add(Protocol(10, 'G', 'probe'))
  session.add(Protocol(11, 'G', 'probe'))
  session.add(Protocol(12, 'G', 'probe'))


def create_tables(verbose):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(location(dbname()), echo=verbose)
  File.metadata.create_all(engine)
  Client.metadata.create_all(engine)
  Session.metadata.create_all(engine)
  Protocol.metadata.create_all(engine)

def create(args):
  """Central creation method."""

  if args.recreate: 
    dbfile = location(dbname()).replace('sqlite:///','')
    if args.verbose and os.path.exists(dbfile):
      print('unlinking %s...' % dbfile)
    if os.path.exists(dbfile): os.unlink(dbfile)

  # the real work...
  create_tables(args.verbose)
  s = session(dbname(), echo=args.verbose)
  add_files(s, args.imagedir)
  add_sessions(s)
  add_protocols(s)
  s.commit()
  s.close()
