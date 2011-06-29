#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 12 May 08:19:50 2011 

"""This script creates the Replay-Attack database in a single pass.
"""

import os

from . import dbname
from ..utils import session, location
from .models import *

def add_clients(session, protodir):
  """Add clients to the replay attack database."""
  
  def add_client_set(session, filename, set):
    """Loads a single client list."""

    for client in open(filename, 'rt'):
      s = client.strip()
      if not s: continue #empty line
      id = int(s)
      session.add(Client(id, set))

  add_client_set(session, os.path.join(protodir, 'client.train'), 'train')
  add_client_set(session, os.path.join(protodir, 'client.devel'), 'devel')
  add_client_set(session, os.path.join(protodir, 'client.test'),  'test')

def add_real_lists(session, protodir):
  """Adds all RCD filelists"""

  def add_real_list(session, filename):
    """Adds an RCD filelist and materializes RealAccess'es."""

    def parse_rcd_filename(f):
      """Parses the RCD filename and break it in the relevant chunks."""

      v = os.path.splitext(os.path.basename(f))[0].split('_')
      client_id = int(v[0].replace('client',''))
      path = os.path.splitext(f)[0] #keep only the filename stem
      purpose = v[3]
      light = v[4]
      take = int(v[5])
      return [client_id, path, light], [purpose, take]

    for fname in open(filename, 'rt'):
      s = fname.strip()
      if not s: continue #emtpy line
      filefields, realfields = parse_rcd_filename(s)
      filefields[0] = session.query(Client).filter(Client.id == filefields[0]).one()
      file = File(*filefields)
      session.add(file)
      realfields.insert(0, file)
      session.add(RealAccess(*realfields))

  add_real_list(session, os.path.join(protodir, 'real.train.list'))
  add_real_list(session, os.path.join(protodir, 'real.devel.list'))
  add_real_list(session, os.path.join(protodir, 'real.test.list'))

def add_attack_lists(session, protodir):
  """Adds all RAD filelists"""

  def add_attack_list(session, filename):
    """Adds an RAD filelist and materializes Attacks."""

    def parse_rad_filename(f):
      """Parses the RAD filename and break it in the relevant chunks."""

      v = os.path.splitext(os.path.basename(f))[0].split('_')
      attack_device = v[1]
      client_id = int(v[2].replace('client',''))
      path = os.path.splitext(f)[0] #keep only the filename stem
      sample_device = v[4]
      sample_type = v[5]
      light = v[6]
      attack_support = f.split('/')[-2]
      return [client_id, path, light], [attack_support, attack_device, sample_type, sample_device]

    for fname in open(filename, 'rt'):
      s = fname.strip()
      if not s: continue #emtpy line
      filefields, attackfields = parse_rad_filename(s)
      filefields[0] = session.query(Client).filter(Client.id == filefields[0]).one()
      file = File(*filefields)
      session.add(file)
      attackfields.insert(0, file)
      session.add(Attack(*attackfields))

  add_attack_list(session,os.path.join(protodir, 'attack.grandtest.train.list'))
  add_attack_list(session,os.path.join(protodir, 'attack.grandtest.devel.list'))
  add_attack_list(session,os.path.join(protodir, 'attack.grandtest.test.list'))

def create_tables(verbose):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(location(dbname()), echo=verbose)
  Client.metadata.create_all(engine)
  RealAccess.metadata.create_all(engine)
  Attack.metadata.create_all(engine)

# Driver API
# ==========

help_message = 'Creates or re-creates this database'

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
  add_clients(s, args.protodir)
  add_real_lists(s, args.protodir)
  add_attack_lists(s, args.protodir)
  s.commit()
  s.close()

def add_commands(parser):
  """Add specific subcommands that the action "create" can use"""

  parser.add_argument('--recreate', action='store_true', default=False,
      help="If set, I'll first erase the current database")
  parser.add_argument('--verbose', action='store_true', default=False,
      help="Do SQL operations in a verbose way")
  parser.add_argument('--protodir', action='store', 
      default='/idiap/group/replay/database/protocols',
      metavar='DIR',
      help="Change the relative path to the directory containing the protocol definitions for replay attacks (defaults to %(default)s)")
  
  parser.set_defaults(func=create) #action
