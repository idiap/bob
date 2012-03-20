#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 28 Jun 2011 15:20:09 CEST 

"""Commands this database can respond to.
"""

import os
import sys

def reverse(args):
  """Returns a list of file database identifiers given the path stems"""

  from .query import Database
  db = Database()

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  r = db.reverse(args.path)
  for id in r: output.write('%d\n' % id)
  
  if not r: sys.exit(1)

def reverse_command(subparsers):
  """Adds the specific options for the reverse command"""

  from argparse import SUPPRESS

  parser = subparsers.add_parser('reverse', help=reverse.__doc__)

  parser.add_argument('path', nargs='+', type=str, help="one or more path stems to look up. If you provide more than one, files which cannot be reversed will be omitted from the output.")
  parser.add_argument('--self-test', dest="selftest", default=False,
      action='store_true', help=SUPPRESS)

  parser.set_defaults(func=reverse) #action

def path(args):
  """Returns a list of fully formed paths or stems given some file id"""

  from .query import Database
  db = Database()

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  r = db.paths(args.id, prefix=args.directory, suffix=args.extension)
  for path in r: output.write('%s\n' % path)

  if not r: sys.exit(1)

def path_command(subparsers):
  """Adds the specific options for the path command"""

  from argparse import SUPPRESS

  parser = subparsers.add_parser('path', help=path.__doc__)

  parser.add_argument('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  parser.add_argument('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  parser.add_argument('id', nargs='+', type=int, help="one or more file ids to look up. If you provide more than one, files which cannot be found will be omitted from the output. If you provide a single id to lookup, an error message will be printed if the id does not exist in the database. The exit status will be non-zero in such case.")
  parser.add_argument('--self-test', dest="selftest", default=False,
      action='store_true', help=SUPPRESS)

  parser.set_defaults(func=path) #action

def add_commands(parser):
  """Adds my subset of options and arguments to the top-level parser. For
  details on syntax, please consult:

  http://docs.python.org/dev/library/argparse.html

  The strategy assumed here is that each command will have its own set of
  options that are relevant to that command. So, we just scan such commands and
  attach the options from those.
  """

  from . import dbname
  from ..utils import location, standard_commands
  from . import __doc__ as dbdoc
  from argparse import RawDescriptionHelpFormatter

  # creates a top-level parser for this database
  myname = dbname()
  top_level = parser.add_parser(myname,
      formatter_class=RawDescriptionHelpFormatter,
      help="Photo/Video Replay attack database", description=dbdoc)
  top_level.set_defaults(dbname=myname)
  top_level.set_defaults(location=location(myname))

  # declare it has subparsers for each of the supported commands
  subparsers = top_level.add_subparsers(title="subcommands")

  # attach standard commands
  standard_commands(subparsers)

  # get the "create" action from a submodule
  from .create import add_command as create_command
  create_command(subparsers)

  # get the "dumplist" action from a submodule
  from .dumplist import add_command as dumplist_command
  dumplist_command(subparsers)

  # get the "checkfiles" action from a submodule
  from .checkfiles import add_command as checkfiles_command
  checkfiles_command(subparsers)

  # adds the "reverse" command
  reverse_command(subparsers)

  # adds the "path" command
  path_command(subparsers)
