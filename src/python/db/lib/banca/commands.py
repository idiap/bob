#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Commands this database can respond to.
"""

import os
import sys

def dbshell(options):
  """Runs a DB interactive shell for the user."""
  
  from ..utils import dbshell as dbs
  from . import dbname

  sys.exit(dbs(dbname()))

def add_commands(parser):
  """Adds my subset of options and arguments to the top-level parser. For
  details on syntax, please consult:

  http://docs.python.org/dev/library/argparse.html

  The strategy assumed here is that each command will have its own set of
  options that are relevant to that command. So, we just scan such commands and
  attach the options from those.
  """

  from . import dbname
  from . import __doc__ as top_doc

  # creates a top-level parser for this database
  top_level = parser.add_parser(dbname(),
      help="BANCA database",
      description=top_doc)

  # declare it has subparsers for each of the supported commands
  subparsers = top_level.add_subparsers(title="subcommands")

  # declare the "dbshell" command, no options for this one
  dbshell_parser = subparsers.add_parser('dbshell', help='Starts a new backend specific shell')
  dbshell_parser.set_defaults(func=dbshell)

  # creates the database from thin-air, accepts the location of the protocol
  # files as input.
  from .create import create
  create_parser = subparsers.add_parser('create', help='Creates or re-creates this database')
  create_parser.add_argument('--recreate', action='store_true', default=False,
      help="If set, I'll first erase the current database")
  create_parser.add_argument('--verbose', action='store_true', default=False,
      help="Do SQL operations in a verbose way")
  create_parser.add_argument('--imagedir', action='store', 
      default='/idiap/group/vision/visidiap/databases/banca/english/images_gray',
      metavar='DIR',
      help="Change the relative path to the directory containing the images of the BANCA database (defaults to %(default)s)")
  create_parser.set_defaults(func=create)

  # dumps lists of filenames according to certain criteria
  from .dumplist import dumplist
  dumplist_parser = subparsers.add_parser('dumplist', help='Dumps lists of files based on your criteria')
  dumplist_parser.add_argument('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  dumplist_parser.add_argument('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  dumplist_parser.add_argument('-p', '--protocol', dest="protocol", default='', help="if given, limits the dump to a particular subset of the data that corresponds to the given protocol (defaults to '%(default)s')", choices=('P', 'G', 'Mc', 'Md', 'Ma', 'Ud', 'Ua', ''))
  dumplist_parser.add_argument('-u', '--purposes', dest="purposes", default='', help="if given, this value will limit the output files to those designed for the given purposes. (defaults to '%(default)s')", choices=('enrol', 'probe', 'world', ''))
  # TODO: client_ids
  dumplist_parser.add_argument('-g', '--groups', dest="groups", default='', help="if given, this value will limit the output files to those belonging to a particular protocolar group. (defaults to '%(default)s')", choices=('g1', 'g2', 'wm', ''))
  dumplist_parser.add_argument('-l', '--languages', dest="langages", default='', help="if given, this value will limit the output files to those belonging to the given languages. (defaults to '%(default)s')", choices=('en', ''))
  dumplist_parser.set_defaults(func=dumplist)

