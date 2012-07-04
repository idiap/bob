#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Wed Jul  4 14:12:51 CEST 2012
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
import tempfile, shutil


def dumplist(args):
  """Dumps lists of files based on your criteria"""

  from .query import Database
  db = Database()

  r = db.files(
      directory=args.directory,
      extension=args.extension,
      groups=args.group,
      protocol=args.protocol,
      purposes=args.purpose,
      sessions=args.session,
      expressions=args.expression,
      illuminations=args.illumination,
      occlusions=args.occlusion)

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  for f in sorted(r.values()):
    output.write('%s\n' % (f,))


def checkfiles(args):
  """Checks existence of files based on your criteria"""

  from .query import Database
  db = Database()

  r = db.files(
      directory=args.directory,
      extension=args.extension)

  # go through all files, check if they are available on the filesystem
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
  from argparse import RawDescriptionHelpFormatter, SUPPRESS

  # creates a top-level parser for this database
  myname = dbname()
  top_level = parser.add_parser(myname,
      formatter_class=RawDescriptionHelpFormatter,
      help="AR face database", description=dbdoc)
  top_level.set_defaults(dbname=myname)
  top_level.set_defaults(location=location(myname))

  # declare it has subparsers for each of the supported commands
  subparsers = top_level.add_subparsers(title="subcommands")

  # attach standard commands
  standard_commands(subparsers)

  # get the "create" action from a submodule
  from .create import add_command as create_command
  from .create import Client, File, Protocol 
  create_command(subparsers)

  # get the "dumplist" action from a submodule
  dump_list_parser = subparsers.add_parser('dumplist', help=dumplist.__doc__)

  dump_list_parser.add_argument('-d', '--directory', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  dump_list_parser.add_argument('-e', '--extension', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  dump_list_parser.add_argument('-g', '--group', help="if given, this value will limit the output files to those belonging to a particular group. (defaults to '%(default)s')", choices = Client.s_groups)
  dump_list_parser.add_argument('-p', '--protocol', default = 'all', help="limits the dump to a particular subset of the data that corresponds to the given protocol (defaults to '%(default)s')", choices = Protocol.s_protocols)
  dump_list_parser.add_argument('-u', '--purpose', help="if given, this value will limit the output files to those designed for the given purposes.", choices=File.s_purposes)
  dump_list_parser.add_argument('-s', '--session', help="if given, this value will limit the output files to those designed for the given session.", choices=File.s_sessions)
  dump_list_parser.add_argument('-w', '--gender', help="if given, this value will limit the output files to those designed for the given gender.", choices=Client.s_genders)
  dump_list_parser.add_argument('-x', '--expression', help="if given, this value will limit the output files to those designed for the given expression.", choices=File.s_expressions)
  dump_list_parser.add_argument('-i', '--illumination', help="if given, this value will limit the output files to those designed for the given illumination.", choices=File.s_illuminations)
  dump_list_parser.add_argument('-o', '--occlusion', help="if given, this value will limit the output files to those designed for the given illumination.", choices=File.s_occlusions)
  dump_list_parser.add_argument('--self-test', dest="selftest", action='store_true', help=SUPPRESS)

  dump_list_parser.set_defaults(func=dumplist) #action

  # get the "checkfiles" action from a submodule
  check_files_parser = subparsers.add_parser('checkfiles', help=checkfiles.__doc__)

  check_files_parser.add_argument('-d', '--directory', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  check_files_parser.add_argument('-e', '--extension', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  check_files_parser.add_argument('--self-test', dest="selftest", action='store_true', help=SUPPRESS)

  check_files_parser.set_defaults(func=checkfiles) #action

