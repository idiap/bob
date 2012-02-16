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

def location(dbname):
  """Returns the location of the database. The location of the database, by
  default, is the directory containing the database library. If the environment
  variable $BOB_DB_DIR is defined, it is used as the base-location of the
  file."""
  
  envvar = 'BOB_DB_DIR'

  if os.environ.has_key(envvar) and os.environ[envvar].strip():
    dirname = os.path.realpath(os.environ[envvar])
    fname = os.path.join(dirname, dbname + '.sql3')

  else:
    dirname = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dirname, dbname, 'db.sql3')

  return 'sqlite:///%s' % fname

def session(dbname, echo=False):
  """Creates a session to an SQLite database"""

  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  engine = create_engine(location(dbname), echo=echo)
  Session = sessionmaker(bind=engine)
  return Session()

def dbshell(options):
  """Drops you into a database shell"""

  import subprocess

  arguments = ['sqlite3', options.location.replace('sqlite:///','')]

  try:
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
  parser.set_defaults(func=dbshell)

def download(options):
  """Downloads the database from a remote server."""

  from .openanything import fetch

  url = '/'.join((options.url.strip('/'), options.version.strip('/'), 
      options.dbname + '.sql3'))
  location = options.location.replace('sqlite:///','')

  # If location exists and it has a .etag file with it, read it and
  # give it to the fetch method
  etag = None
  if os.path.exists(location) and os.path.exists(location + '.etag'):
    etag = open(location + '.etag', 'rt').read().strip()

  if options.verbose: 
    print "Requesting %s" % (url,)
 
  data = fetch(url, etag=etag)

  if data['status'] == 200:
    output = open(location, 'wb')
    output.write(data['data'])
    output.close()

    if options.verbose:
      print "Gzip Compression: %s" % data['gzip']
      print "Database Size: %d bytes" % len(data['data'])
      print "Last Modification: %s" % data['lastmodified']
      print "Saved at: %s" % location

    if data['etag']:
      if options.verbose: 
        print "E-Tag: %s" % data['etag']
      etag_file = open(location + '.etag', 'wt')
      etag_file.write(data['etag'])
      etag_file.close()
      print "E-Tag cached: %s" % (location + '.etag',)

  elif data['status'] == 304: #etag matches
    if options.verbose:
      print "Currently installed version is up-to-date (did not re-download)"

  else:
    raise IOError, "Failed download of %s (status: %d)" % (url, data['status'])

def download_command(subparsers):
  """Adds a new download subcommand to your subparser"""
  
  DOWNLOAD_URL = 'http://www.idiap.ch/software/bob/databases/'
  """Location from where to download bob databases"""

  DOWNLOAD_VERSION = 'nightlies/last'
  """The default version to use for the databases"""

  parser = subparsers.add_parser('download', help=download.__doc__)
  parser.add_argument('-u', '--url', dest="url", default=DOWNLOAD_URL, help="changes the url from where to download the database files (defaults to '%(default)s')")
  parser.add_argument('-v', '--version', dest="version", default=DOWNLOAD_VERSION, help="changes the base version of the databases to download (defaults to '%(default)s')")
  parser.add_argument('--verbose', dest="verbose", default=False,
      action='store_true', help="produces more output while downloading")
  parser.set_defaults(func=download)

  return parser

def print_location(options):
  """Prints the current location of the database SQL file."""
  
  if options.with_protocol:
    print(options.location)
  else:
    print(options.location.replace('sqlite:///',''))

def location_command(subparsers):
  """Adds a new location subcommand to your parser"""

  parser = subparsers.add_parser('location', help=print_location.__doc__)
  parser.add_argument('--with-protocol', dest="with_protocol", 
      default=False, action='store_true', 
      help="prints the filepath or directory leading to the database with the specific database protocol prepended")
  parser.set_defaults(func=print_location)

  return parser

def copy(options):
  """Copies the database to a given directory."""

  import shutil
  d = options.directory[0]
  if not os.path.exists(d): os.makedirs(d)
  dest = os.path.join(d, options.dbname + '.sql3')
  if os.path.exists(dest): os.unlink(dest)
  shutil.copy2(options.location.replace('sqlite:///',''), dest)

def copy_command(subparsers):
  
  parser = subparsers.add_parser('copy', help=copy.__doc__)
  parser.add_argument('directory', help="sets the directory to which the database will be copied to", nargs=1)
  parser.set_defaults(func=copy)

  return parser

def standard_commands(subparsers):
  """Adds all standard commands to databases that can respond to them."""

  dbshell_command(subparsers)
  download_command(subparsers)
  location_command(subparsers)
  copy_command(subparsers)

def makedirs_safe(fulldir):
  """Creates a directory if it does not exists, with concurrent access support"""
  try:
    if not os.path.exists(fulldir): os.makedirs(fulldir)
  except OSError as exc: # Python >2.5
    if exc.errno == errno.EEXIST: pass
    else: raise

def check_group_writeability(filename):
  """Applies group writeability if the directory containing the database has
  the write bit set, or at least tries to."""

  # extracts the current mode of the parent directory
  mode = os.stat(os.path.dirname(filename)).st_mode

  # if the mode of the directory sets group to be writeable, try to apply this
  # to the given file as well.
  if (mode & stat.S_IWGRP):
    try:
      os.chmod(filename, mode | stat.S_IWGRP)
    except OSError, ex:
      logging.warn("Cannot make '%s' group-writeable despite the fact its parent directory is group-writeable. This maybe a problem depending on your database setup. Please check." % filename)
