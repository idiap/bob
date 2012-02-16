#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 12 May 08:19:50 2011 

"""This script creates the Replay-Attack database in a single pass.
"""

import os
import fnmatch

from .models import *
from ..utils import session

def add_clients(session, protodir, verbose):
  """Add clients to the replay attack database."""
  
  def add_client_set(session, filename, set):
    """Loads a single client list."""

    for client in open(filename, 'rt'):
      s = client.strip()
      if not s: continue #empty line
      id = int(s)
      if verbose: print "Adding client %d on '%s' set..." % (id, set)
      session.add(Client(id, set))

  add_client_set(session, os.path.join(protodir, 'client.train'), 'train')
  add_client_set(session, os.path.join(protodir, 'client.devel'), 'devel')
  add_client_set(session, os.path.join(protodir, 'client.test'),  'test')

def add_real_lists(session, protodir, verbose):
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
      if len(v) == 6: take = int(v[5]) #authentication session
      else: take = 1 #enrollment session
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
  add_real_list(session, os.path.join(protodir, 'enrollment.list'))

def add_attack_lists(session, protodir, verbose):
  """Adds all RAD filelists"""

  def add_attack_list(session, filename):
    """Adds an RAD filelist and materializes Attacks."""

    def parse_rad_filename(f):
      """Parses the RAD filename and break it in the relevant chunks."""

      v = os.path.splitext(os.path.basename(f))[0].split('_')
      attack_device = v[1] #print, mobile or highdef
      client_id = int(v[2].replace('client',''))
      path = os.path.splitext(f)[0] #keep only the filename stem
      sample_device = v[4] #highdef or mobile
      sample_type = v[5] #photo or video
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

def define_protocols(session, protodir, verbose):
  """Defines all available protocols"""

  #figures out which protocols to use
  valid = set()
  for fname in fnmatch.filter(os.listdir(protodir), 'real.*.train.list'):
    s = fname.split('.', 3)
    consider = True
    for cls in ('real', 'attack'):
      for grp in ('train', 'devel', 'test'):
        search_for = os.path.join(protodir, '%s.%s.%s.list' % (cls, s[1], grp))
        if not os.path.exists(search_for):
          if verbose:
            print "Not considering protocol %s as '%s' was not found" % \
              (s[1], search_for)
          consider = False
    if consider: valid.add(s[1])

  for protocol in valid:
    if verbose: print "Creating protocol '%s'..." % protocol

    # create protocol on the protocol table
    obj = Protocol(name=protocol)

    for grp in ('train', 'devel', 'test'):

      flist = os.path.join(protodir, 'real.%s.%s.list' % (protocol, grp))
      counter = 0
      for fname in open(flist, 'rt'):
        s = os.path.splitext(fname.strip())[0]
        q = session.query(RealAccess).join(File).filter(File.path == s).one()
        q.protocols.append(obj)
        counter += 1
      if verbose: print "  -> %5s/%-6s: %d files" % (grp, "real", counter)
      
      counter = 0
      flist = os.path.join(protodir, 'attack.%s.%s.list' % (protocol, grp))
      for fname in open(flist, 'rt'):
        s = os.path.splitext(fname.strip())[0]
        q = session.query(Attack).join(File).filter(File.path == s).one()
        q.protocols.append(obj)
        counter += 1
      if verbose: print "  -> %5s/%-6s: %d files" % (grp, "attack", counter)
   
    session.add(obj)

def create_tables(args):
  """Creates all necessary tables (only to be used at the first time)"""

  from sqlalchemy import create_engine
  engine = create_engine(args.location, echo=args.verbose)
  Client.metadata.create_all(engine)
  RealAccess.metadata.create_all(engine)
  Attack.metadata.create_all(engine)
  Protocol.metadata.create_all(engine)

# Driver API
# ==========

def create(args):
  """Creates or re-creates this database"""

  dbfile = args.location.replace('sqlite:///','')

  args.verbose = 0 if args.verbose is None else sum(args.verbose)

  if args.recreate: 
    if args.verbose and os.path.exists(dbfile):
      print('unlinking %s...' % dbfile)
    if os.path.exists(dbfile): os.unlink(dbfile)

  if not os.path.exists(os.path.dirname(dbfile)):
    os.makedirs(os.path.dirname(dbfile))

  # the real work...
  create_tables(args)
  s = session(args.dbname, echo=(args.verbose >= 2))
  add_clients(s, args.protodir, args.verbose)
  add_real_lists(s, args.protodir, args.verbose)
  add_attack_lists(s, args.protodir, args.verbose)
  define_protocols(s, args.protodir, args.verbose)
  s.commit()
  s.close()
  
def add_command(subparsers):
  """Add specific subcommands that the action "create" can use"""

  parser = subparsers.add_parser('create', help=create.__doc__)

  parser.add_argument('-R', '--recreate', action='store_true', default=False,
      help="If set, I'll first erase the current database")
  parser.add_argument('-v', '--verbose', action='append_const', const=1, 
      help="Do SQL operations in a verbose way")
  parser.add_argument('-D', '--protodir', action='store', 
      default='/idiap/group/replay/database/protocols',
      metavar='DIR',
      help="Change the relative path to the directory containing the protocol definitions for replay attacks (defaults to %(default)s)")
  
  parser.set_defaults(func=create) #action
