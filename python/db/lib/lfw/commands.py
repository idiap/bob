#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
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
      help="LFW database", description=dbdoc)
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
