#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Wed Jul  4 14:12:51 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""This script creates the AR face database in a single pass.
"""

import os

from .models import *
from ..utils import session


def add_clients(session):
  """Adds the clients and split up the groups 'world', 'dev', and 'eval'"""
  # these are the ids that contain less than 26 images; 20 out of 136 identities
  # and some other ids, to round up 50 world identities
  world_ids_m = set((8,11,24,28,29,34,35,50,57,62, 63,64,68,    1,4,9,10,16,22,26, 37,40,42,49,54,70,72,76))
  world_ids_w = set((1,6,10,27,47,49,56,    3,7,14, 16,20,24,26,33,34,36,40,41,44,52,55)) 
  
  dev_ids_m = set((2,5,6,7,13,15,18,21,23,25,27,30,33,38,44,45,48,52,53,59,66,69,73,75))
  dev_ids_w = set((4,9,12,13,17,18,23,28,29,31,37,38,43,46,48,51,54,58,59))

  # assert that world and dev set are independent
  assert len(world_ids_m) + len(dev_ids_m) == len(world_ids_m | dev_ids_m)
  assert len(world_ids_w) + len(dev_ids_w) == len(world_ids_w | dev_ids_w)

  eval_ids_m = set(range(1,77)) - world_ids_m - dev_ids_m
  eval_ids_w = set(range(1,61)) - world_ids_w - dev_ids_w
  

  # now, we have: world: 28 male, 22 female; dev and eval: 24 male and 19 female
  # add these clients
  for id in world_ids_m:
    session.add(Client("m-%03d"%id, 'world'))
  for id in world_ids_w:
    session.add(Client("w-%03d"%id, 'world'))

  for id in dev_ids_m:
    session.add(Client("m-%03d"%id, 'dev'))
  for id in dev_ids_w:
    session.add(Client("w-%03d"%id, 'dev'))

  for id in eval_ids_m:
    session.add(Client("m-%03d"%id, 'eval'))
  for id in eval_ids_w:
    session.add(Client("w-%03d"%id, 'eval'))

def add_files(session, directory, extension):
  """Adds the files from the given directory"""
  files = os.listdir(directory)
  for file in files:
    parts = os.path.splitext(file) 
    if parts[1] == extension:
      session.add(File(parts[0]))

def add_protocols(session):
  """Adds various protocols for the AR face database"""
  for s in File.s_sessions:
    # different expressions
    for e in File.s_expressions[1:]:
      session.add(Protocol('all', s, expression=e))
      session.add(Protocol('expression', s, expression=e))

    # different illuminations
    for i in File.s_illuminations[1:]:
      session.add(Protocol('all', s, illumination=i))
      session.add(Protocol('illumination', s, illumination=i))

    for o in File.s_occlusions[1:]:
      session.add(Protocol('all', s, occlusion=o))
      session.add(Protocol('occlusion', s, occlusion=o))
      
      # add mixed occlusion/illumination protocol
      for i in File.s_illuminations[1:3]:
        session.add(Protocol('all', s, illumination=i, occlusion=o))
        session.add(Protocol('occlusion_and_illumination', s, illumination=i, occlusion=o))

    # add the neutral files to all of the protocols
    session.add(Protocol('all', s, expression=File.s_expressions[0], illumination=File.s_illuminations[0], occlusion=File.s_occlusions[0]))
    session.add(Protocol('expression', s, expression=File.s_expressions[0], illumination=File.s_illuminations[0], occlusion=File.s_occlusions[0]))
    session.add(Protocol('illumination', s, expression=File.s_expressions[0], illumination=File.s_illuminations[0], occlusion=File.s_occlusions[0]))
    session.add(Protocol('occlusion', s, expression=File.s_expressions[0], illumination=File.s_illuminations[0], occlusion=File.s_occlusions[0]))
    session.add(Protocol('occlusion_and_illumination', s, expression=File.s_expressions[0], illumination=File.s_illuminations[0], occlusion=File.s_occlusions[0]))
    

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
  add_files(s, args.directory, args.extension)
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
  parser.add_argument('--directory', metavar='DIR',
      default="/idiap/resource/database/AR_Face/images",
      help="The path to the images of the AR face database")
  parser.add_argument('--extension', metavar='STR', default='.ppm',
      help="The file extension of the image files from the AR face database")
  
  parser.set_defaults(func=create) #action
