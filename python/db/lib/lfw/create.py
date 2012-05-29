#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date: Thu May 24 10:41:42 CEST 2012
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

"""This script creates the Labeled Faces in the Wild (LFW) database in a single pass.
"""

import os

from .models import *
from ..utils import session


def nodot(item):
  """Can be used to ignore hidden files, starting with the . character."""
  return item[0] != '.'

def add_files(session, basedir):
  """Adds files to the LFW database.
     Returns dictionaries with ids of the clients and ids of the files 
     in the generated SQL tables"""

  def add_client(session, client_id):
    """Adds a client to the LFW database."""
    c = Client(client_dir)
    session.add(c)
 
  def add_file(session, file_name):
    """Parses a single filename and add it to the list."""
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    shot_id = base_name.split('_')[-1]
    client_id = base_name[0:-len(shot_id)-1]
    f = File(client_id, shot_id)
    session.add(f)

  # Loops over the directory structure
  imagedir = os.path.join(basedir, 'all_images')
  for client_dir in filter(nodot, sorted([d for d in os.listdir(imagedir)])):
    # adds a client to the database
    client_name = add_client(session, client_dir)
    for filename in filter(nodot, sorted([d for d in os.listdir(os.path.join(imagedir, client_dir))])):
      if filename.endswith('.jpg'):
        # adds a file to the database
        file_id = add_file(session, filename )
  

def add_people(session, basedir):
  """Adds the people to the LFW database"""

  def add_client(session, protocol, client_id, count):
    """Adds all images of a client"""
    for i in range(1,count+1):
      session.add(People(protocol, File(client_id, i).m_id))

  def parse_view1(session, filename, protocol):
    """Parses a file containing the people of view 1 of the LFW database"""
    pfile = open(filename)
    for line in pfile:
      llist = line.split()
      if len(llist) == 2: # one person and the number of images 
        add_client(session, protocol, llist[0], int(llist[1]))

  def parse_view2(session, filename):
    """Parses the file containing the people of view 2 of the LFW database"""
    fold_id = 0
    pfile = open(filename)
    for line in pfile:
      llist = line.split()
      if len(llist) == 1: # the number of persons in the list
        protocol = "fold"+str(fold_id)
        fold_id += 1
      elif len(llist) == 2: # one person and the number of images 
        add_client(session, protocol, llist[0], int(llist[1]))
    

  # Adds view1 people
  parse_view1(session, os.path.join(basedir, 'view1', 'peopleDevTrain.txt'), 'train')
  parse_view1(session, os.path.join(basedir, 'view1', 'peopleDevTest.txt'), 'test')

  # Adds view2 people
  parse_view2(session, os.path.join(basedir, 'view2', 'people.txt'))

def add_pairs(session, basedir):
  """Adds the pairs for all protocols of the LFW database"""

  def add_mpair(session, protocol, file_id1, file_id2):
    """Add a matched pair to the LFW database."""
    session.add(Pair(protocol, file_id1, file_id2, True))

  def add_upair(session, protocol, file_id1, file_id2):
    """Add an unmatched pair to the LFW database."""
    session.add(Pair(protocol, file_id1, file_id2, False))

  def parse_file(session, filename, protocol):
    """Parses a file containing pairs and adds them to the LFW database"""
    pfile = open(filename)
    for line in pfile:
      llist = line.split()
      if len(llist) == 3: # Matched pair 
        file_id1 = File(llist[0], int(llist[1])).m_id
        file_id2 = File(llist[0], int(llist[2])).m_id
        add_mpair(session, protocol, file_id1, file_id2)
      elif len(llist) == 4: # Unmatched pair
        file_id1 = File(llist[0], int(llist[1])).m_id
        file_id2 = File(llist[2], int(llist[3])).m_id
        add_upair(session, protocol, file_id1, file_id2)

  # Adds view1 pairs
  parse_file(session, os.path.join(basedir, 'view1', 'pairsDevTrain.txt'), 'train')
  parse_file(session, os.path.join(basedir, 'view1', 'pairsDevTest.txt'), 'test')

  # Adds view2 pairs
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold1.txt'), 'fold1')
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold2.txt'), 'fold2')
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold3.txt'), 'fold3')
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold4.txt'), 'fold4')
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold5.txt'), 'fold5')
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold6.txt'), 'fold6')
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold7.txt'), 'fold7')
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold8.txt'), 'fold8')
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold9.txt'), 'fold9')
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold10.txt'), 'fold10')


def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  Client.metadata.create_all(engine)
  File.metadata.create_all(engine)
  People.metadata.create_all(engine)
  Pair.metadata.create_all(engine)

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
  add_files(s, args.basedir)
  add_people(s, args.basedir)
  add_pairs(s, args.basedir)
  s.commit()
  s.close()

def add_command(subparsers):
  """Add specific subcommands that the action "create" can use"""

  parser = subparsers.add_parser('create', help=create.__doc__)

  parser.add_argument('--recreate', action='store_true', default=False,
      help="If set, I'll first erase the current database")
  parser.add_argument('--verbose', action='store_true', default=False,
      help="Do SQL operations in a verbose way")
  parser.add_argument('--basedir', action='store', metavar='DIR',
      default='/idiap/resource/database/lfw',
      help="Change the relative path to the directory containing the images of the LFW database (defaults to %(default)s)")
  
  parser.set_defaults(func=create) #action
