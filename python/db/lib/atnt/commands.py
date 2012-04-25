#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date: Fri Apr 20 12:04:44 CEST 2012
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

"""Commands (at command line) for the AT&T database.
"""

import os
import sys

def dumplist(args):
  """Dumps lists of files based on your criteria."""

  from .__init__ import Database
  db = Database()

  r = db.files(directory=args.directory, extension=args.extension, groups=args.groups, purposes=args.purposes)

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  for id, f in r.items():
    output.write('%s\n' % (f,))
  

def checkfiles(args):
  """Checks the existence of the files based on your criteria.""" 
    
  from .__init__ import Database
  db = Database()

  r = db.files(directory=args.directory, extension=args.extension)

  # go through all files, check if they are available
  good = {}
  bad = {}
  for id, f in r.items():
    if os.path.exists(f): good[id] = f
    else: bad[id] = f

  # report
  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  if bad:
    for id, f in bad.items():
      output.write('Cannot find file "%s"\n' % (f,))
    output.write('%d files (out of %d) were not found at "%s"\n' % \
        (len(bad), len(r), args.directory))



def add_commands(parser):
  """Add the command line options supported by the AT&T database. 
  """
  
  # creates a top-level parser for this database
  from . import Database
  db = Database()
  myname = db.dbname()

  import argparse

  # creates a top-level parser for this database
  top_level = parser.add_parser(db.dbname(),
      formatter_class=argparse.RawDescriptionHelpFormatter,
      help="AT&T Database",
      description=__doc__)

  # declare it has subparsers for each of the supported commands
  subparsers = top_level.add_subparsers(title="subcommands")

  # add the dumplist command
  dump_parser = subparsers.add_parser('dumplist', help="Dumps list of files based on your criteria")
  dump_parser.add_argument('-d', '--directory', default=None, help="if given, this path will be prepended to every entry returned")
  dump_parser.add_argument('-e', '--extension', default=None, help="if given, this extension will be appended to every entry returned")
  dump_parser.add_argument('-g', '--groups', default=None, help="if given, this value will limit the output files to those belonging to a particular group.", choices=db.m_groups)
  dump_parser.add_argument('-p', '--purposes', default=None, help="if given, this value will limit the output files to those belonging to a particular purpose.", choices=db.m_purposes)
  dump_parser.add_argument('--self-test', dest="selftest", action='store_true', help=argparse.SUPPRESS)
  dump_parser.set_defaults(func=dumplist) #action

  # add the checkfiles command
  check_parser = subparsers.add_parser('checkfiles', help="Check if the files exist, based on your criteria")
  check_parser.add_argument('-d', '--directory', required=True, help="The path to the AT&T images")
  check_parser.add_argument('-e', '--extension', default=".pgm", help="The extension of the AT&T images default: '.pgm'")
  check_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=argparse.SUPPRESS)
  check_parser.set_defaults(func=checkfiles) #action
