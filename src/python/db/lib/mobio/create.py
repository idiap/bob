#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This script creates the MOBIO database in a single pass.
"""

import os

from .models import *
from ..utils import session


def nodot(item):
  """Can be used to ignore hidden files, starting with the . character."""
  return item[0] != '.'

def add_files(session, imagedir):
  """Add files to the MOBIO database."""
 
  def add_file(session, basename):
    """Parse a single filename and add it to the list.
       Also add a client entry if not already in the database."""
    v = os.path.splitext(basename)[0].split('_')
    bname = os.path.splitext(basename)[0]
    gender = v[0][0]
    institute = int(v[0][1])
    if institute == 0:
      institute = 'idiap'
    elif institute == 1:
      institute = 'manchester'
    elif institute == 2:
      institute = 'surrey'
    elif institute == 3:
      institute = 'oulu'
    elif institute == 4:
      institute = 'brno'
    elif institute == 5:
      institute = 'avignon'
    client_id = v[0][1:4]
    
    if not (client_id in client_dict):
      if (institute == 'surrey' or institute == 'avignon'):
        group = 'world'
      elif (institute == 'manchester' or institute == 'oulu'):
        group = 'dev'
      elif (institute == 'idiap' or institute == 'brno'):
        group = 'eval'
      session.add(Client(int(client_id), group, gender, institute))
      client_dict[client_id] = True

    session_id = int(v[1])
    speech_type = v[2][0]
    shot_id = v[2][1:3]
    environment = v[3][0]
    device = v[3][1]
    if( device == '0'):
      device = 'mobile'
    elif( device == '1'):
      device = 'laptop'
    channel = int(v[4][0])

    session.add(File(int(client_id), bname, session_id, speech_type, shot_id, environment, device, channel))

  client_dict = {} 
  for filename in filter(nodot, os.listdir(imagedir)):
    if filename.endswith('.jpg'):
      add_file(session, os.path.basename(filename) )

def add_protocols(session):
  """Adds protocols"""

  # Protocols: speech types used for a given purpose
  session.add(Protocol('male', 'enrol', 'p'))
  session.add(Protocol('male', 'probe', 'r'))
  session.add(Protocol('male', 'probe', 'f'))
  session.add(Protocol('female', 'enrol', 'p'))
  session.add(Protocol('female', 'probe', 'r'))
  session.add(Protocol('female', 'probe', 'f'))

  # Protocols: Session used for enrolling
  # male: protcol, client_id, session_id for enroling
  session.add(ProtocolEnrolSession('male', 1, 1))
  session.add(ProtocolEnrolSession('male', 2, 1))
  session.add(ProtocolEnrolSession('male', 4, 1))
  session.add(ProtocolEnrolSession('male', 8, 1))
  session.add(ProtocolEnrolSession('male', 11, 1))
  session.add(ProtocolEnrolSession('male', 12, 1))
  session.add(ProtocolEnrolSession('male', 15, 1))
  session.add(ProtocolEnrolSession('male', 16, 1))
  session.add(ProtocolEnrolSession('male', 17, 1))
  session.add(ProtocolEnrolSession('male', 19, 2)) # exception
  session.add(ProtocolEnrolSession('male', 21, 1))
  session.add(ProtocolEnrolSession('male', 23, 1))
  session.add(ProtocolEnrolSession('male', 24, 1))
  session.add(ProtocolEnrolSession('male', 25, 1))
  session.add(ProtocolEnrolSession('male', 26, 1))
  session.add(ProtocolEnrolSession('male', 28, 1))
  session.add(ProtocolEnrolSession('male', 29, 1))
  session.add(ProtocolEnrolSession('male', 30, 1))
  session.add(ProtocolEnrolSession('male', 31, 1))
  session.add(ProtocolEnrolSession('male', 33, 1))
  session.add(ProtocolEnrolSession('male', 34, 1))
  session.add(ProtocolEnrolSession('male', 103, 1))
  session.add(ProtocolEnrolSession('male', 104, 1))
  session.add(ProtocolEnrolSession('male', 106, 1))
  session.add(ProtocolEnrolSession('male', 107, 1))
  session.add(ProtocolEnrolSession('male', 108, 1))
  session.add(ProtocolEnrolSession('male', 109, 1))
  session.add(ProtocolEnrolSession('male', 111, 1))
  session.add(ProtocolEnrolSession('male', 112, 1))
  session.add(ProtocolEnrolSession('male', 114, 1))
  session.add(ProtocolEnrolSession('male', 115, 1))
  session.add(ProtocolEnrolSession('male', 116, 1))
  session.add(ProtocolEnrolSession('male', 117, 1))
  session.add(ProtocolEnrolSession('male', 119, 1))
  session.add(ProtocolEnrolSession('male', 120, 1))
  session.add(ProtocolEnrolSession('male', 301, 1))
  session.add(ProtocolEnrolSession('male', 304, 1))
  session.add(ProtocolEnrolSession('male', 305, 1))
  session.add(ProtocolEnrolSession('male', 308, 1))
  session.add(ProtocolEnrolSession('male', 310, 1))
  session.add(ProtocolEnrolSession('male', 313, 1))
  session.add(ProtocolEnrolSession('male', 314, 1))
  session.add(ProtocolEnrolSession('male', 315, 1))
  session.add(ProtocolEnrolSession('male', 317, 1))
  session.add(ProtocolEnrolSession('male', 319, 1))
  session.add(ProtocolEnrolSession('male', 416, 1))
  session.add(ProtocolEnrolSession('male', 417, 1))
  session.add(ProtocolEnrolSession('male', 418, 1))
  session.add(ProtocolEnrolSession('male', 419, 1))
  session.add(ProtocolEnrolSession('male', 420, 1))
  session.add(ProtocolEnrolSession('male', 421, 1))
  session.add(ProtocolEnrolSession('male', 422, 1))
  session.add(ProtocolEnrolSession('male', 423, 1))
  session.add(ProtocolEnrolSession('male', 424, 1))
  session.add(ProtocolEnrolSession('male', 425, 1))
  session.add(ProtocolEnrolSession('male', 426, 1))
  session.add(ProtocolEnrolSession('male', 427, 1))
  session.add(ProtocolEnrolSession('male', 428, 1))
  session.add(ProtocolEnrolSession('male', 429, 1))
  session.add(ProtocolEnrolSession('male', 430, 1))
  session.add(ProtocolEnrolSession('male', 431, 1))
  session.add(ProtocolEnrolSession('male', 432, 1))
  # female: protcol, client_id, session_id for enroling
  session.add(ProtocolEnrolSession('female', 7, 2)) # exception
  session.add(ProtocolEnrolSession('female', 9, 1))
  session.add(ProtocolEnrolSession('female', 10, 1))
  session.add(ProtocolEnrolSession('female', 22, 1))
  session.add(ProtocolEnrolSession('female', 32, 1))
  session.add(ProtocolEnrolSession('female', 118, 1))
  session.add(ProtocolEnrolSession('female', 122, 1))
  session.add(ProtocolEnrolSession('female', 123, 1))
  session.add(ProtocolEnrolSession('female', 125, 1))
  session.add(ProtocolEnrolSession('female', 126, 1))
  session.add(ProtocolEnrolSession('female', 127, 1))
  session.add(ProtocolEnrolSession('female', 128, 1))
  session.add(ProtocolEnrolSession('female', 129, 1))
  session.add(ProtocolEnrolSession('female', 130, 1))
  session.add(ProtocolEnrolSession('female', 131, 1))
  session.add(ProtocolEnrolSession('female', 133, 1))
  session.add(ProtocolEnrolSession('female', 302, 1))
  session.add(ProtocolEnrolSession('female', 303, 1))
  session.add(ProtocolEnrolSession('female', 306, 1))
  session.add(ProtocolEnrolSession('female', 307, 1))
  session.add(ProtocolEnrolSession('female', 309, 1))
  session.add(ProtocolEnrolSession('female', 311, 1))
  session.add(ProtocolEnrolSession('female', 320, 1))
  session.add(ProtocolEnrolSession('female', 401, 1))
  session.add(ProtocolEnrolSession('female', 402, 1))
  session.add(ProtocolEnrolSession('female', 403, 1))
  session.add(ProtocolEnrolSession('female', 404, 1))
  session.add(ProtocolEnrolSession('female', 405, 2)) # exception
  session.add(ProtocolEnrolSession('female', 406, 1))
  session.add(ProtocolEnrolSession('female', 407, 1))
  session.add(ProtocolEnrolSession('female', 408, 1))
  session.add(ProtocolEnrolSession('female', 409, 1))
  session.add(ProtocolEnrolSession('female', 410, 1))
  session.add(ProtocolEnrolSession('female', 411, 1))
  session.add(ProtocolEnrolSession('female', 412, 1))
  session.add(ProtocolEnrolSession('female', 413, 1))
  session.add(ProtocolEnrolSession('female', 415, 1))
  session.add(ProtocolEnrolSession('female', 433, 1))

def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  Client.metadata.create_all(engine)
  File.metadata.create_all(engine)
  Protocol.metadata.create_all(engine)
  ProtocolEnrolSession.metadata.create_all(engine)

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
      default='/idiap/user/njohan/__mobio/images/selected-images/',
      help="Change the relative path to the directory containing the images of the MOBIO database (defaults to %(default)s)")
  
  parser.set_defaults(func=create) #action
