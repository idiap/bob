#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Fri Jul  6 16:45:41 CEST 2012
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

"""Commands the FRGC database can respond to.
"""

import os
import sys
import tempfile, shutil


def dumplist(args):
  """Dumps lists of files based on your criteria"""

  from .query import Database
  db = Database(args.database)

  r = db.files(
      directory=args.directory,
      extension=args.extension,
      groups=args.groups,
      protocol=args.protocol,
      purposes=args.purposes,
      mask_type = 'maskII') # here we take mask II since this is the combination of mask I and mask III

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  for id, f in r.items():
    output.write('%s\n' % (f,))


def checkfiles(args):
  """Checks existence of files based on your criteria"""

  from .query import Database
  db = Database(args.database)

  r = db.files(
      directory=args.directory,
      extension=args.extension,
      groups=args.groups,
      protocol=args.protocol,
      purposes=args.purposes,
      mask_type = 'maskII') # here we take mask II since this is the combination of mask I and mask III

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


def create_annotation_files(args):
  """Creates the position files for the FRGC database 
  (using the positions stored in the xml files), 
  so that FRGC position files share the same structure as the image files."""  

  # report
  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()
    args.directory = tempfile.mkdtemp(prefix='bob_db_frgc_')
  
  from .query import Database
  db = Database(args.database)
  
  # retrieve all files
  annotations = db.annotations(directory=args.directory, extension=args.extension)
  for annotation in annotations.itervalues():
    filename = annotation[0]
    if not os.path.exists(os.path.dirname(filename)):
      os.makedirs(os.path.dirname(filename))
    positions = annotation[1]
    f = open(filename, 'w')
    # write eyes in the common order: left eye, right eye
    
    for type in ('reye', 'leye', 'nose', 'mouth'):
      f.writelines(type + ' ' + str(positions[type][1]) + ' ' + str(positions[type][0]) + '\n')
    f.close()
    
  
  if args.selftest:
    # check that all files really exist
    args.selftest = False
    args.groups = None
    args.purposes = None
    for args.protocol in ('2.0.1','2.0.2','2.0.4'):
      checkfiles(args)
    shutil.rmtree(args.directory)


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
      help="FRGC database", description=dbdoc)
  top_level.set_defaults(dbname=myname)
  top_level.set_defaults(location=location(myname))

  # declare it has subparsers for each of the supported commands
  subparsers = top_level.add_subparsers(title="subcommands")

  # attach standard commands
  standard_commands(subparsers)

  # get the "dumplist" action from a submodule
  dump_list_parser = subparsers.add_parser('dumplist', help=dumplist.__doc__)

  dump_list_parser.add_argument('-D', '--database', default='/idiap/resource/database/frgc/FRGC-2.0-dist', help="The base directory of the FRGC database")
  dump_list_parser.add_argument('-d', '--directory', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  dump_list_parser.add_argument('-e', '--extension', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  dump_list_parser.add_argument('-g', '--groups', help="if given, this value will limit the output files to those belonging to a particular group. (defaults to '%(default)s')", choices=('world', 'dev'))
  dump_list_parser.add_argument('-p', '--protocol', default = '2.0.1', help="if given, limits the dump to a particular subset of the data that corresponds to the given protocol (defaults to '%(default)s')", choices=('2.0.1', '2.0.2', '2.0.4'))
  dump_list_parser.add_argument('-u', '--purposes', help="if given, this value will limit the output files to those designed for the given purposes. (defaults to '%(default)s')", choices=('enrol', 'probe'))
  dump_list_parser.add_argument('--self-test', dest="selftest", action='store_true', help=SUPPRESS)

  dump_list_parser.set_defaults(func=dumplist) #action

  # get the "checkfiles" action from a submodule
  check_files_parser = subparsers.add_parser('checkfiles', help=checkfiles.__doc__)

  check_files_parser.add_argument('-D', '--database', default='/idiap/resource/database/frgc/FRGC-2.0-dist', help="The base directory of the FRGC database")
  check_files_parser.add_argument('-d', '--directory', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  check_files_parser.add_argument('-e', '--extension', default='.jpg', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  check_files_parser.add_argument('-g', '--groups', help="if given, this value will limit the output files to those belonging to a particular group. (defaults to '%(default)s')", choices=('world', 'dev'))
  check_files_parser.add_argument('-p', '--protocol', default='2.0.1', help="if given, limits the dump to a particular subset of the data that corresponds to the given protocol (defaults to '%(default)s')", choices=('2.0.1', '2.0.2', '2.0.4'))
  check_files_parser.add_argument('-u', '--purposes', help="if given, this value will limit the output files to those designed for the given purposes. (defaults to '%(default)s')", choices=('enrol', 'probe'))
  check_files_parser.add_argument('--self-test', dest="selftest", action='store_true', help=SUPPRESS)

  check_files_parser.set_defaults(func=checkfiles) #action

  # get the "create-eye-files" action from a submodule
  create_annotation_files_parser = subparsers.add_parser('create-annotation-files', help=create_annotation_files.__doc__)

  create_annotation_files_parser.add_argument('-D', '--database', default='/idiap/resource/database/frgc/FRGC-2.0-dist', help="The base directory of the FRGC database")
  create_annotation_files_parser.add_argument('-d', '--directory', required=True, help="The eye position files will be stored in this directory")
  create_annotation_files_parser.add_argument('-e', '--extension', default = '.pos', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  create_annotation_files_parser.add_argument('--self-test', dest="selftest", action='store_true', help=SUPPRESS)

  create_annotation_files_parser.set_defaults(func=create_annotation_files) #action
