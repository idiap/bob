#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Commands this database can respond to.
"""

import os
import sys

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
      help="SCFace database", description=dbdoc)
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
