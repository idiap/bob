#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed 6 Jul 20:58:23 2011 

"""This script creates the XM2VTS database in a single pass.
"""

import os

from .models import *
from ..utils import session


def nodot(item):
  """Can be used to ignore hidden files, starting with the . character."""
  return item[0] != '.'

def add_clients(session):
  """Add clients to the XM2VTS database."""
  # clients
  client_list = [  3,   4,   5,   6,   9,  12,  13,  16,  17,  18, 
                  19,  20,  21,  22,  24,  25,  26,  27,  29,  30, 
                  32,  33,  34,  35,  36,  37,  38,  40,  41,  42,
                  45,  47,  49,  50,  51,  52,  53,  54,  55,  56, 
                  58,  60,  61,  64,  65,  66,  68,  69,  71,  72, 
                  73,  74,  75,  78,  79,  80,  82,  85,  89,  90,
                  91,  92,  99, 101, 102, 103, 105, 108, 110, 112, 
                 113, 115, 116, 121, 122, 123, 124, 125, 126, 129, 
                 132, 133, 135, 136, 137, 138, 140, 141, 145, 146, 
                 148, 150, 152, 154, 159, 163, 164, 165, 166, 167, 
                 168, 169, 173, 178, 179, 180, 181, 182, 183, 188,
                 191, 193, 196, 197, 198, 206, 207, 208, 209, 210, 
                 211, 213, 216, 218, 219, 221, 222, 224, 227, 228, 
                 229, 231, 232, 233, 235, 236, 237, 240, 243, 244, 
                 246, 248, 249, 253, 255, 258, 259, 261, 264, 266, 
                 267, 269, 270, 274, 275, 278, 279, 281, 282, 285, 
                 287, 288, 289, 290, 292, 293, 295, 305, 310, 312, 
                 316, 319, 320, 321, 322, 324, 325, 328, 329, 330, 
                 332, 333, 334, 336, 337, 338, 339, 340, 342, 357, 
                 358, 359, 360, 362, 364, 365, 366, 369, 370, 371] 
  for cid in client_list:
    session.add(Client(cid, 'client'))
  # impostorDev 
  impostorDev_list = [  0,   2,   7,  46,  57,  62,  83,  93, 104, 120, 
                      143, 157, 158, 177, 187, 189, 203, 212, 215, 242, 
                      276, 284, 301, 314, 331]
  for cid in impostorDev_list:
    session.add(Client(cid, 'impostorDev'))
  # impostorEval 
  impostorEval_list = [  1,   8,  10,  11,  23,  28,  31,  39,  43,  44, 
                        48,  59,  67,  70,  81,  86,  87,  88,  95,  96,
                        98, 107, 109, 111, 114, 119, 127, 128, 130, 131, 
                       134, 142, 147, 149, 153, 155, 160, 161, 170, 171, 
                       172, 174, 175, 176, 185, 190, 199, 200, 201, 202, 
                       225, 226, 234, 241, 250, 263, 271, 272, 280, 283, 
                       286, 300, 313, 315, 317, 318, 323, 335, 341, 367]
  for cid in impostorEval_list:
    session.add(Client(cid, 'impostorEval'))
                      
def add_files(session, imagedir):
  """Add files to the XM2VTS database."""
 
  def add_file(session, basename, client_dir, subdir):
    """Parse a single filename and add it to the list."""
    v = os.path.splitext(basename)[0].split('_')
    if(subdir == 'frontal'):
      session.add(File(int(v[0]), os.path.join(subdir, client_dir, basename), int(v[1]), 'n', int(v[2])))  
    elif(subdir == 'darkened'):
      session.add(File(int(v[0]), os.path.join(subdir, client_dir, basename), 4, v[2][0], int(v[2][1])))  
 
  for subdir in ('frontal', 'darkened'): 
    imagedir_app = os.path.join(imagedir,subdir)
    file_list = os.listdir(imagedir_app)
    for cl_dir in filter(nodot, file_list):
      if os.path.isdir(os.path.join(imagedir_app, cl_dir)):
        client_dir = os.path.join(imagedir_app, cl_dir)
        for filename in filter(nodot, os.listdir(client_dir)):
          basename, extension = os.path.splitext(filename)
          add_file(session, basename, cl_dir, subdir)

def add_protocols(session):
  """Adds protocols"""
  # Protocol lp1
  session.add(Protocol('lp1', '', 'enrol', 1, 'n', 1))
  session.add(Protocol('lp1', '', 'enrol', 2, 'n', 1))
  session.add(Protocol('lp1', '', 'enrol', 3, 'n', 1))
  session.add(Protocol('lp1', 'dev', 'probe', 1, 'n', 2))
  session.add(Protocol('lp1', 'dev', 'probe', 2, 'n', 2))
  session.add(Protocol('lp1', 'dev', 'probe', 3, 'n', 2))
  session.add(Protocol('lp1', 'eval', 'probe', 4, 'n', 1))
  session.add(Protocol('lp1', 'eval', 'probe', 4, 'n', 2))
  
  # Protocol lp2
  session.add(Protocol('lp2', '', 'enrol', 1, 'n', 1))
  session.add(Protocol('lp2', '', 'enrol', 1, 'n', 2))
  session.add(Protocol('lp2', '', 'enrol', 2, 'n', 1))
  session.add(Protocol('lp2', '', 'enrol', 2, 'n', 2))
  session.add(Protocol('lp2', 'dev', 'probe', 3, 'n', 1))
  session.add(Protocol('lp2', 'dev', 'probe', 3, 'n', 2))
  session.add(Protocol('lp2', 'eval', 'probe', 4, 'n', 1))
  session.add(Protocol('lp2', 'eval', 'probe', 4, 'n', 2))

  # Protocol darkened-lp1
  session.add(Protocol('darkened-lp1', '', 'enrol', 1, 'n', 1))
  session.add(Protocol('darkened-lp1', '', 'enrol', 2, 'n', 1))
  session.add(Protocol('darkened-lp1', '', 'enrol', 3, 'n', 1))
  session.add(Protocol('darkened-lp1', 'dev', 'probe', 1, 'n', 2))
  session.add(Protocol('darkened-lp1', 'dev', 'probe', 2, 'n', 2))
  session.add(Protocol('darkened-lp1', 'dev', 'probe', 3, 'n', 2))
  session.add(Protocol('darkened-lp1', 'eval', 'probe', 4, 'l', 1))
  session.add(Protocol('darkened-lp1', 'eval', 'probe', 4, 'l', 2))
  session.add(Protocol('darkened-lp1', 'eval', 'probe', 4, 'r', 1))
  session.add(Protocol('darkened-lp1', 'eval', 'probe', 4, 'r', 2))

  # Protocol darkened-lp2
  session.add(Protocol('darkened-lp2', '', 'enrol', 1, 'n', 1))
  session.add(Protocol('darkened-lp2', '', 'enrol', 1, 'n', 2))
  session.add(Protocol('darkened-lp2', '', 'enrol', 2, 'n', 1))
  session.add(Protocol('darkened-lp2', '', 'enrol', 2, 'n', 2))
  session.add(Protocol('darkened-lp2', 'dev', 'probe', 3, 'n', 1))
  session.add(Protocol('darkened-lp2', 'dev', 'probe', 3, 'n', 2))
  session.add(Protocol('darkened-lp2', 'eval', 'probe', 4, 'l', 1))
  session.add(Protocol('darkened-lp2', 'eval', 'probe', 4, 'l', 2))
  session.add(Protocol('darkened-lp2', 'eval', 'probe', 4, 'r', 1))
  session.add(Protocol('darkened-lp2', 'eval', 'probe', 4, 'r', 2))


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
      default='/idiap/resource/database/xm2vtsdb/images/',
      help="Change the relative path to the directory containing the images of the XM2VTS database (defaults to %(default)s)")
  
  parser.set_defaults(func=create) #action
