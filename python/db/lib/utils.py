#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 12 May 08:33:24 2011 

"""Some utilities to talk to bob SQLite databases.
"""

import os
import sys
import errno
import stat
import logging

class null(object):
  """A look-alike stream that discards the input"""

  def write(self, s):
    pass
  
  def flush(self):
    pass

def session(dbtype, dbdir, dbfile, echo=False):
  """Creates a session to an SQLite database"""

  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  engine = create_engine(connection_string(dbtype, dbdir, dbfile), echo=echo)
  Session = sessionmaker(bind=engine)
  return Session()

def connection_string(dbtype, dbdir, dbfile):
  """Returns a connection string for supported platforms"""

  if dbtype == 'sqlite': return 'sqlite:///' + os.path.join(dbdir, dbfile)
    
  # otherwise, we just raise
  raise RuntimeError, "Cannot create connection strings for database types different than 'sqlite' and you have type='%s'" % (dbtype,)

def dbshell(options):
  """Drops you into a database shell"""

  import subprocess

  dbfile = os.path.join(options.location, options.files[0])

  if options.type == 'sqlite': 
    prog = 'sqlite3'
  else: 
    raise RuntimeError, "Error auxiliary database file '%s' cannot be used to initiate a database shell connection (type='%s')" % (dbfile, options.type)

  arguments = [prog, dbfile]

  try:
    if options.dryrun:
      print "[dry-run] exec '%s'" % ' '.join(arguments)
      return 0
    else:
      p = subprocess.Popen(arguments)
  except OSError as e:
    # occurs when the file is not executable or not found
    print("Error executing '%s': %s (%d)" % (' '.join(arguments), e.strerror,
        e.errno))
    sys.exit(e.errno)
  
  try:
    p.communicate()
  except KeyboardInterrupt: # the user CTRL-C'ed
    import signal
    os.kill(p.pid, signal.SIGTERM)
    return signal.SIGTERM

  return p.returncode

def dbshell_command(subparsers):
  """Adds a new dbshell subcommand to your subparser"""

  parser = subparsers.add_parser('dbshell', help=dbshell.__doc__)
  parser.add_argument("-n", "--dry-run", dest="dryrun", default=False,
      action='store_true',
      help="does not actually run, just prints what would do instead")
  parser.set_defaults(func=dbshell)

def print_location(options):
  """Prints the current location of the database SQL file."""
  
  for k in options.files: print(os.path.join(options.location, k))

  return 0

def location_command(subparsers):
  """Adds a new location subcommand to your parser"""

  parser = subparsers.add_parser('location', help=print_location.__doc__)
  parser.set_defaults(func=print_location)

  return parser

def get(options):
  """Copies the database auxiliary files from a given directory to their installed working location."""

  from .download import download
  import shutil

  for f in options.files:
    src = '/'.join((options.url, options.dbname, options.version, f))
    dest = os.path.join(options.location, f)

    if not os.path.exists(options.location):
      if options.verbose: 
        print "Creating directory '%s'..." % options.location

      makedirs_safe(options.location, options.dryrun)

    if os.path.exists(dest):
      if options.verbose: print "Removing existing copy '%s'..." % dest

      if options.dryrun:
        print "[dry-run] rm -f '%s'" % dest
      else:
        os.unlink(dest)

    if options.verbose: print "Getting %s -> %s" % (src, dest)

    if src.find('http') == 0: #must download
      if options.dryrun: 
        print "[dry-run] wget --output-document=%s %s" % (dest, src)
      else: 
        download(src, dest, options.verbose)
    else:
      if options.dryrun: 
        print "[dry-run] cp %s %s" % (src, dest)
      else: 
        shutil.copy2(src, dest)

  return 0

def get_command(subparsers):
  
  parser = subparsers.add_parser('get', help=get.__doc__)
  parser.add_argument("-n", "--dry-run", dest="dryrun", default=False,
      action='store_true',
      help="does not actually run, just prints what would do instead")
  parser.add_argument('-v', '--verbose', dest="verbose", default=False,
      action='store_true', help="produces more output while copying")
  parser.add_argument('-V', '--version', dest="version",
      help="if set, overrides the default version set for this package when putting the database files on the given directory")
  parser.add_argument('url', default="http://www.idiap.ch/software/bob/databases", 
      help="sets the URL to which the database will be downloaded from (defaults to '%(default)s')",
      nargs="?")
  parser.set_defaults(func=get)

  return parser

def put(options):
  """Copies database auxiliary files from their official work location to a given directory."""

  import shutil

  d = os.path.join(options.directory, options.dbname, options.version)
  makedirs_safe(d, options.dryrun)
  for f in options.files:
    dest = os.path.join(d, f)

    if os.path.exists(dest):
      if options.verbose: print "Removing existing file '%s'..." % dest
      
      if options.dryrun:
        print "[dry-run] rm -f '%s'" % dest
      else:
        os.unlink(dest)

    src = os.path.join(options.location, f)
    
    if options.verbose: print "Putting %s -> %s..." % (src, dest)

    if options.dryrun: print "[dry-run] cp %s -> %s" % (src, dest)
    else: shutil.copy2(src, dest)

  return 0

def put_command(subparsers):
  
  parser = subparsers.add_parser('put', help=put.__doc__)
  parser.add_argument("-n", "--dry-run", dest="dryrun", default=False,
      action='store_true',
      help="does not actually run, just prints what would do instead")
  parser.add_argument('-v', '--verbose', dest="verbose", default=False,
      action='store_true', help="produces more output while copying")
  parser.add_argument('-V', '--version', dest="version",
      help="if set, overrides the default version set for this package when putting the database files on the given directory")
  parser.add_argument('directory', help="sets the directory to which the database will be copied to")
  parser.set_defaults(func=put)

  return parser

def standard_commands(subparsers, type, files):
  """Adds all standard commands to databases that can respond to them."""

  if type == 'sqlite': 
    dbshell_command(subparsers)

  if files:
    location_command(subparsers)
    put_command(subparsers)
    get_command(subparsers)

def makedirs_safe(fulldir, dryrun=False):
  """Creates a directory if it does not exists, with concurrent access support"""
  try:
    if dryrun:
      print "[dry-run] mkdir -p '%s'" % fulldir
    else:
      if not os.path.exists(fulldir): os.makedirs(fulldir)

  except OSError as exc: # Python >2.5
    if exc.errno == errno.EEXIST: pass
    else: raise
