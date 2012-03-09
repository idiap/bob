#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This script creates the SCFace database in a single pass.
"""

import os

from .models import *
from ..utils import session


def nodot(item):
  """Can be used to ignore hidden files, starting with the . character."""
  return item[0] != '.'

def add_clients(session, filename):
  """Add clients to the SCFace database."""

  # open features.txt file containing information about the clients
  f = open(filename, 'r')
  c = 0
  for line in f:
    # Ignore the 10 first (useless) lines
    c = c + 1
    if c<=10:
      continue
    
    # parse the line
    tok = line.split('\t')

    # birthyear
    birthyear = tok[1].split('.')[2]
    # group
    if int(tok[0]) <= 43:
      group = 'world'
    elif int(tok[0]) <= 87:
      group = 'dev'
    else:
      group = 'eval'
    # gender
    if int(tok[2]) == 0:
      gender = 'm'
    else:
      gender = 'f'

    # Add the client
    session.add(Client(int(tok[0]), group, int(birthyear), gender, int(tok[3]), int(tok[4]), int(tok[5])))

def add_subworlds(session):
  """Adds splits in the world set, based on the client ids"""
  # one third
  session.add(Subworld("onethird", 1))
  session.add(Subworld("onethird", 4))
  session.add(Subworld("onethird", 5))
  session.add(Subworld("onethird", 6))
  session.add(Subworld("onethird", 8))
  session.add(Subworld("onethird", 11))
  session.add(Subworld("onethird", 12))
  session.add(Subworld("onethird", 18))
  session.add(Subworld("onethird", 20))
  session.add(Subworld("onethird", 30))
  session.add(Subworld("onethird", 33))
  session.add(Subworld("onethird", 36))
  session.add(Subworld("onethird", 39))
  session.add(Subworld("onethird", 40))
  
  # two thirds
  session.add(Subworld("twothirds", 2))
  session.add(Subworld("twothirds", 3))
  session.add(Subworld("twothirds", 7))
  session.add(Subworld("twothirds", 9))
  session.add(Subworld("twothirds", 10))
  session.add(Subworld("twothirds", 13))
  session.add(Subworld("twothirds", 14))
  session.add(Subworld("twothirds", 15))
  session.add(Subworld("twothirds", 16))
  session.add(Subworld("twothirds", 17))
  session.add(Subworld("twothirds", 19))
  session.add(Subworld("twothirds", 21))
  session.add(Subworld("twothirds", 22))
  session.add(Subworld("twothirds", 23))
  session.add(Subworld("twothirds", 24))
  session.add(Subworld("twothirds", 25))
  session.add(Subworld("twothirds", 26))
  session.add(Subworld("twothirds", 27))
  session.add(Subworld("twothirds", 28))
  session.add(Subworld("twothirds", 29))
  session.add(Subworld("twothirds", 31))
  session.add(Subworld("twothirds", 32))
  session.add(Subworld("twothirds", 34))
  session.add(Subworld("twothirds", 35))
  session.add(Subworld("twothirds", 37))
  session.add(Subworld("twothirds", 38))
  session.add(Subworld("twothirds", 41))
  session.add(Subworld("twothirds", 42))
  session.add(Subworld("twothirds", 43))

 
def add_files(session, imagedir):
  """Add files to the SCFace database."""
 
  def add_file(session, basename, maindir, frontal):
    """Parse a single filename and add it to the list."""
    v = os.path.splitext(basename)[0].split('_')
    if frontal:
      session.add(File(int(v[0]), os.path.join(maindir, basename), 'frontal', 0))
    else:
      session.add(File(int(v[0]), os.path.join(maindir, basename), v[1], int(v[2])))

  for maindir in ['mugshot_frontal_cropped_all', 'surveillance_cameras_distance_1',\
                  'surveillance_cameras_distance_2', 'surveillance_cameras_distance_3']:
    if not os.path.isdir( os.path.join( imagedir, maindir) ):
      continue
    elif maindir == 'mugshot_frontal_cropped_all':
      for f in filter(nodot, os.listdir( os.path.join( imagedir, maindir) )):
        basename, extension = os.path.splitext(f)
        add_file(session, basename, maindir, True)
    else:
      for camdir in filter(nodot, os.listdir( os.path.join( imagedir, maindir) )):
        subdir = os.path.join(maindir, camdir)
        for f in filter(nodot, os.listdir( os.path.join( imagedir, subdir) )):
          basename, extension = os.path.splitext(f)
          add_file(session, basename, subdir, False)
   

def add_protocols(session):
  """Adds protocols"""

  # Protocols
  session.add(Protocol('combined', 'enrol', 'frontal', 0))
  session.add(Protocol('combined', 'probe', 'cam1', 1))
  session.add(Protocol('combined', 'probe', 'cam1', 2))
  session.add(Protocol('combined', 'probe', 'cam1', 3))
  session.add(Protocol('combined', 'probe', 'cam2', 1))
  session.add(Protocol('combined', 'probe', 'cam2', 2))
  session.add(Protocol('combined', 'probe', 'cam2', 3))
  session.add(Protocol('combined', 'probe', 'cam3', 1))
  session.add(Protocol('combined', 'probe', 'cam3', 2))
  session.add(Protocol('combined', 'probe', 'cam3', 3))
  session.add(Protocol('combined', 'probe', 'cam4', 1))
  session.add(Protocol('combined', 'probe', 'cam4', 2))
  session.add(Protocol('combined', 'probe', 'cam4', 3))
  session.add(Protocol('combined', 'probe', 'cam5', 1))
  session.add(Protocol('combined', 'probe', 'cam5', 2))
  session.add(Protocol('combined', 'probe', 'cam5', 3))
  session.add(Protocol('close', 'enrol', 'frontal', 0))
  session.add(Protocol('close', 'probe', 'cam1', 3))
  session.add(Protocol('close', 'probe', 'cam2', 3))
  session.add(Protocol('close', 'probe', 'cam3', 3))
  session.add(Protocol('close', 'probe', 'cam4', 3))
  session.add(Protocol('close', 'probe', 'cam5', 3))
  session.add(Protocol('medium', 'enrol', 'frontal', 0))
  session.add(Protocol('medium', 'probe', 'cam1', 2))
  session.add(Protocol('medium', 'probe', 'cam2', 2))
  session.add(Protocol('medium', 'probe', 'cam3', 2))
  session.add(Protocol('medium', 'probe', 'cam4', 2))
  session.add(Protocol('medium', 'probe', 'cam5', 2))
  session.add(Protocol('far', 'enrol', 'frontal', 0))
  session.add(Protocol('far', 'probe', 'cam1', 1))
  session.add(Protocol('far', 'probe', 'cam2', 1))
  session.add(Protocol('far', 'probe', 'cam3', 1))
  session.add(Protocol('far', 'probe', 'cam4', 1))
  session.add(Protocol('far', 'probe', 'cam5', 1))
  
def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  Client.metadata.create_all(engine)
  Subworld.metadata.create_all(engine)
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
  add_clients(s, args.featuresfile)
  add_subworlds(s)
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
  parser.add_argument('--featuresfile', action='store', metavar='FILE',
      default='/idiap/resource/database/scface/SCface_database/features.txt',
      help="Change the path to the file containing information about the clients of the SCFace database (defaults to %(default)s)") 
  parser.add_argument('--imagedir', action='store', metavar='DIR',
      default='/idiap/group/biometric/databases/scface/images',
      help="Change the relative path to the directory containing the images of the SCFace database (defaults to %(default)s)")
  
  parser.set_defaults(func=create) #action
