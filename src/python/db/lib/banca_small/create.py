#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri 20 May 17:00:50 2011 

"""This script creates the BANCA_SMALL database in a single pass.
"""

import os

from .models import *
from ..utils import session

def add_files(session):
  """Add files (and clients) to the BANCA_SMALL database."""
 
  def add_file(session, filename, client_dict):
    """Parse a single filename and add it to the list.
       Also add a client entry if not already in the database."""

    v = os.path.splitext(os.path.basename(filename))[0].split('_')
    if not (v[0] in client_dict):
      if (v[2] == 'wm'):
        v[2] = 'world'
      session.add(Client(int(v[0]), v[1], v[2], v[5]))
      client_dict[v[0]] = True
    session_id = int(v[3].split('s')[1])
    session.add(File(int(v[0]), os.path.basename(filename).split('.')[0], v[4], v[6], session_id))
  
  file_list = []
  file_list.append('9003_m_wm_s01_9003_en_1')
  file_list.append('9003_m_wm_s01_9003_en_2')
  file_list.append('9003_m_wm_s01_9003_en_3')
  file_list.append('9003_m_wm_s01_9003_en_4')
  file_list.append('9003_m_wm_s01_9003_en_5')
  file_list.append('9003_m_wm_s01_9004_en_1')
  file_list.append('9003_m_wm_s01_9004_en_2')
  file_list.append('9003_m_wm_s01_9004_en_3')
  file_list.append('9003_m_wm_s01_9004_en_4')
  file_list.append('9003_m_wm_s01_9004_en_5')
  file_list.append('9005_m_wm_s01_9005_en_1')
  file_list.append('9005_m_wm_s01_9005_en_2')
  file_list.append('9005_m_wm_s01_9005_en_3')
  file_list.append('9005_m_wm_s01_9005_en_4')
  file_list.append('9005_m_wm_s01_9005_en_5')
  file_list.append('9005_m_wm_s01_9006_en_1')
  file_list.append('9005_m_wm_s01_9006_en_2')
  file_list.append('9005_m_wm_s01_9006_en_3')
  file_list.append('9005_m_wm_s01_9006_en_4')
  file_list.append('9005_m_wm_s01_9006_en_5')
  file_list.append('9007_m_wm_s01_9007_en_1')
  file_list.append('9007_m_wm_s01_9007_en_2')
  file_list.append('9007_m_wm_s01_9007_en_3')
  file_list.append('9007_m_wm_s01_9007_en_4')
  file_list.append('9007_m_wm_s01_9007_en_5')
  file_list.append('9007_m_wm_s01_9008_en_1')
  file_list.append('9007_m_wm_s01_9008_en_2')
  file_list.append('9007_m_wm_s01_9008_en_3')
  file_list.append('9007_m_wm_s01_9008_en_4')
  file_list.append('9007_m_wm_s01_9008_en_5')
  file_list.append('9009_m_wm_s01_9009_en_1')
  file_list.append('9009_m_wm_s01_9009_en_2')
  file_list.append('9009_m_wm_s01_9009_en_3')
  file_list.append('9009_m_wm_s01_9009_en_4')
  file_list.append('9009_m_wm_s01_9009_en_5')
  file_list.append('9009_m_wm_s01_9010_en_1')
  file_list.append('9009_m_wm_s01_9010_en_2')
  file_list.append('9009_m_wm_s01_9010_en_3')
  file_list.append('9009_m_wm_s01_9010_en_4')
  file_list.append('9009_m_wm_s01_9010_en_5')
  file_list.append('9011_m_wm_s01_9011_en_1')
  file_list.append('9011_m_wm_s01_9011_en_2')
  file_list.append('9011_m_wm_s01_9011_en_3')
  file_list.append('9011_m_wm_s01_9011_en_4')
  file_list.append('9011_m_wm_s01_9011_en_5')
  file_list.append('9011_m_wm_s01_9012_en_1')
  file_list.append('9011_m_wm_s01_9012_en_2')
  file_list.append('9011_m_wm_s01_9012_en_3')
  file_list.append('9011_m_wm_s01_9012_en_4')
  file_list.append('9011_m_wm_s01_9012_en_5')
  file_list.append('1001_f_g1_s01_1001_en_1')
  file_list.append('1001_f_g1_s01_1001_en_2')
  file_list.append('1001_f_g1_s01_1001_en_3')
  file_list.append('1001_f_g1_s01_1001_en_4')
  file_list.append('1001_f_g1_s01_1001_en_5')
  file_list.append('1001_f_g1_s02_1001_en_1')
  file_list.append('1001_f_g1_s02_1001_en_2')
  file_list.append('1002_f_g1_s12_1001_en_1')
  file_list.append('1002_f_g1_s12_1001_en_2')

  client_dict = {} 
  for filename in file_list:
    add_file(session, filename, client_dict)

def add_protocols(session):
  """Adds protocols"""
  # Protocol P
  session.add(Protocol(1, 'P', 'enrol'))
  session.add(Protocol(2, 'P', 'probe'))
  session.add(Protocol(12, 'P', 'probe'))

def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  File.metadata.create_all(engine)
  Client.metadata.create_all(engine)
  Protocol.metadata.create_all(engine)

# Driver API
# ==========

def create(args):
  """Creates or re-creates this database"""

  dbfile = args.location.replace('sqlite:///','')

  if args.recreate: 
    if args.verbose and os.path.exists(dbfile):
      print('unlinking %s...' % dbfile)
    if os.path.exists(dbfile): os.unlink(dbfile)

  if not os.path.exists(os.path.dirname(dbfile)):
    os.makedirs(os.path.dirname(dbfile))

  # the real work...
  create_tables(args)
  s = session(args.dbname, echo=args.verbose)
  add_files(s)
  add_protocols(s)
  s.commit()
  s.close()

def add_command(subparsers):
  """Add specific subcommands that the action "create" can use"""

  parser = subparsers.add_parser('create', help=create.__doc__)

  parser.add_argument('--recreate', action='store_true', default=False,
      help="If set, I'll first erase the current database")
  parser.add_argument('--verbose', action='store_true', default=False,
      help="Do SQL operations in a verbose way")
  
  parser.set_defaults(func=create) #action
