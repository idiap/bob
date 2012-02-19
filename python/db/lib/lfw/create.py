#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""This script creates the LFW database in a single pass.
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

  def add_client(session, client_dir):
    """Adds a client to the LFW database."""
    c = Client(client_dir)
    session.add(c)
    # We want to make use of the new assigned client id
    # We need to do the following:
    session.flush()
    session.refresh(c)
    return c.id
 
  def add_file(session, c_id, client_dir, basename):
    """Parses a single filename and add it to the list."""
    bname = os.path.splitext(basename)[0]
    shot = bname.split('_')[-1]
    f = File(int(c_id), os.path.join(client_dir, bname), int(shot))
    session.add(f)
    # We want to make use of the new assigned file id
    # We need to do the following:
    session.flush()
    session.refresh(f)
    return (f.id, int(shot))

  # Loops over the directory structure
  client_dict = {} # dict[client_name_string] = client_id_sql
  file_dict = {} # dict[(client_id_sql,file_shot_id)] = file_id_sql
  imagedir = os.path.join(basedir, 'all_images')
  for client_dir in filter(nodot, sorted([d for d in os.listdir(imagedir)])):
    # adds a client to the database
    c_id = add_client(session, client_dir)
    client_dict[client_dir] = c_id
    for filename in filter(nodot, sorted([d for d in os.listdir(os.path.join(imagedir, client_dir))])):
      if filename.endswith('.jpg'):
        # adds a file to the database
        (f_id, f_shot_id) = add_file(session, c_id, client_dir, os.path.basename(filename) )
        file_dict[(c_id, f_shot_id)] = f_id
  
  return (client_dict, file_dict)

def add_pairs(session, basedir, client_dict, file_dict):
  """Adds pairs"""

  def add_mpair(session, view, subset, client_id1, shot_id1, shot_id2):
    """Add a matched pair to the LFW database."""
    session.add(Pair(view, subset, client_id1, shot_id1, client_id1, shot_id2))

  def add_upair(session, view, subset, client_id1, shot_id1, client_id2, shot_id2):
    """Add an unmatched pair to the LFW database."""
    session.add(Pair(view, subset, client_id1, shot_id1, client_id2, shot_id2))

  def parse_file(session, filename, view, subset, file_dict):
    """Parses a file containing pairs and adds them to the LFW database"""
    pfile = open(filename)
    for line in pfile:
      llist = line.split()
      if len(llist) == 3: # Matched pair 
        f_id1 = file_dict[(client_dict[llist[0]], int(llist[1]))]
        f_id2 = file_dict[(client_dict[llist[0]], int(llist[2]))]
        add_mpair(session, view, subset, client_dict[llist[0]], f_id1, f_id2)
      elif len(llist) == 4: # Unmatched pair
        f_id1 = file_dict[(client_dict[llist[0]], int(llist[1]))]
        f_id2 = file_dict[(client_dict[llist[2]], int(llist[3]))]
        add_upair(session, view, subset, client_dict[llist[0]], f_id1, client_dict[llist[2]], f_id2)

  # Adds view1 pairs
  parse_file(session, os.path.join(basedir, 'view1', 'pairsDevTrain.txt'), 'view1', 'train', file_dict)
  parse_file(session, os.path.join(basedir, 'view1', 'pairsDevTest.txt'), 'view1', 'test', file_dict)

  # Adds view2 pairs
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold1.txt'), 'view2', 'fold1', file_dict)
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold2.txt'), 'view2', 'fold2', file_dict)
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold3.txt'), 'view2', 'fold3', file_dict)
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold4.txt'), 'view2', 'fold4', file_dict)
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold5.txt'), 'view2', 'fold5', file_dict)
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold6.txt'), 'view2', 'fold6', file_dict)
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold7.txt'), 'view2', 'fold7', file_dict)
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold8.txt'), 'view2', 'fold8', file_dict)
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold9.txt'), 'view2', 'fold9', file_dict)
  parse_file(session, os.path.join(basedir, 'view2', 'pairs_fold10.txt'), 'view2', 'fold10', file_dict)


def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  Client.metadata.create_all(engine)
  File.metadata.create_all(engine)
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
  (client_dict, file_dict) = add_files(s, args.basedir)
  add_pairs(s, args.basedir, client_dict, file_dict)
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
