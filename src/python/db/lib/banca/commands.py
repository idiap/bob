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
  dbshell_parser = subparsers.add_parser('dbshell',
      help='Starts a new backend specific shell')
  dbshell_parser.set_defaults(func=dbshell)

  # get the "create" action from a submodule
  from .create import add_commands as create_commands
  from .create import help_message as create_message
  create_parser = subparsers.add_parser('create', help=create_message)
  create_commands(create_parser)

  # get the "dumplist" action from a submodule
  from .dumplist import add_commands as dumplist_commands
  from .dumplist import help_message as dumplist_message
  dumplist_parser = subparsers.add_parser('dumplist', help=dumplist_message)
  dumplist_commands(dumplist_parser)
