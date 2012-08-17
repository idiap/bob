#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed 6 Jul 20:58:23 2011 

"""This script creates the FIR database in a single pass.
"""

import os

from .models import *
from ..utils import session


def nodot(item):
  """Can be used to ignore hidden files, starting with the . character."""
  return item[0] != '.'

def add_clients(session):
  """Add clients to the FIR database."""
  # clients
  session.add(Client(1, 'dev'))
  session.add(Client(2, 'world'))
  session.add(Client(3, 'dev'))
  session.add(Client(4, 'world'))
  session.add(Client(5, 'dev'))
  session.add(Client(6, 'dev'))
  session.add(Client(7, 'world'))
  session.add(Client(8, 'dev'))
  session.add(Client(9, 'world'))
  session.add(Client(10, 'world'))
  session.add(Client(11, 'dev'))
  session.add(Client(12, 'world'))
  session.add(Client(13, 'eval'))
  session.add(Client(15, 'eval'))
  session.add(Client(16, 'world'))
  session.add(Client(17, 'world'))
  session.add(Client(18, 'eval'))
  session.add(Client(19, 'eval'))
  session.add(Client(20, 'eval'))
  session.add(Client(21, 'world'))
  session.add(Client(22, 'eval'))

def add_files(session, imagedir):
  """Add files to the FIR database."""
 
  def add_file(session, basename):
    """Parse a single filename and add it to the list."""
    v = os.path.splitext(basename)[0].split('_')
    ir = True
    if v[1] == '0':
      ir = False
    else:
      ir = True
    session.add(File(int(v[0]), basename, ir, int(v[2]), int(v[3]), int(v[4])))
  
  file_list = os.listdir(imagedir)
  for filename in filter(nodot, file_list):
    basename, extension = os.path.splitext(filename)
    add_file(session, basename)

def add_protocols(session):
  """Adds protocols"""
  for illumination in range(1,7):
    for shot in range(0,5):
      if illumination == 2:
        session.add(Protocol('ir', 'world', 'enrol', True, illumination, shot))
        session.add(Protocol('noir', 'world', 'enrol', False, illumination, shot))
        session.add(Protocol('ir', 'dev', 'enrol', True, illumination, shot))
        session.add(Protocol('noir', 'dev', 'enrol', False, illumination, shot))
        session.add(Protocol('ir', 'eval', 'enrol', True, illumination, shot))
        session.add(Protocol('noir', 'eval', 'enrol', False, illumination, shot))
      else:
        session.add(Protocol('ir', 'world', 'probe', True, illumination, shot))
        session.add(Protocol('noir', 'world', 'probe', False, illumination, shot))
        session.add(Protocol('ir', 'dev', 'probe', True, illumination, shot))
        session.add(Protocol('noir', 'dev', 'probe', False, illumination, shot))
        session.add(Protocol('ir', 'eval', 'probe', True, illumination, shot))
        session.add(Protocol('noir', 'eval', 'probe', False, illumination, shot))
  

def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  Client.metadata.create_all(engine)
  File.metadata.create_all(engine)
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
  add_clients(s)
  add_files(s, args.imagedir)
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
  parser.add_argument('--imagedir', action='store', metavar='DIR',
      default='/idiap/group/biometric/databases/fir/db/',
      help="Change the relative path to the directory containing the images of the FIR database (defaults to %(default)s)")
  
  parser.set_defaults(func=create) #action
