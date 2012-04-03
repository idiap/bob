#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Sun Mar 25 18:33:12 CEST 2012

"""Dumps lists of files.
"""

import os
import sys

# Driver API
# ==========

def dumplist(args):
  """Dumps lists of files based on your criteria"""

  from .__init__ import Database
  db = Database()

  r = db.files(directory=args.directory, extension=args.extension, groups=args.group, cls=args.cls, versions=args.version)

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  for id, f in r.items():
    output.write('%s\n' % (f,))
  
def dumpfold(args):
  """Dumps lists of files belonging to a certain cross validation fold based on your criteria"""

  from .__init__ import Database
  db = Database()

  r = db.cross_valid_foldfiles(version=args.version, cls=args.cls, fold_no=args.fold_no, directory=args.directory, extension=args.extension)
  
  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  if args.subset == 'validation':
    for id, f in r[0].items():
      output.write('%s\n' % (f,))

  if args.subset == 'training':
    for id, f in r[1].items():
      output.write('%s\n' % (f,))

  if args.subset == '':
    output.write("Validations subset:\n")
    for id, f in r[0].items():
      output.write('%s\n' % (f,))
    output.write("Training subset:\n")
    for id, f in r[1].items():
      output.write('%s\n' % (f,))


def checkfiles(args):
  """Checks the existence of the files based on your criteria""" 
    
  from .__init__ import Database
  db = Database()

  r = db.files(directory=args.directory, extension=args.extension, groups=args.group, cls=args.cls, versions=args.version)

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
  """Add specific subcommands that the action "dumplist" can use"""

  from argparse import SUPPRESS
  from .__init__ import Database
  from ..utils import location

  db = Database()

  # from . import dbname
  from argparse import RawDescriptionHelpFormatter, SUPPRESS

  # creates a top-level parser for this database
  top_level = parser.add_parser(db.dbname(),
      formatter_class=RawDescriptionHelpFormatter,
      help="NUAA Face Spoofing Database",
      description=__doc__)

  # declare it has subparsers for each of the supported commands
  subparsers = top_level.add_subparsers(title="subcommands")

  # add the dumplist command
  dump_message = "Dumps list of files based on your criteria"
  dump_parser = subparsers.add_parser('dumplist', help=dump_message)
  dump_parser.add_argument('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  dump_parser.add_argument('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  dump_parser.add_argument('-c', '--class', dest="cls", default=None, help="if given, limits the dump to a particular subset of the data that corresponds to the given class (defaults to '%(default)s')", choices=db.classes)
  dump_parser.add_argument('-g', '--group', dest="group", default=None, help="if given, this value will limit the output files to those belonging to a particular group. (defaults to '%(default)s')", choices=db.groups)
  dump_parser.add_argument('-v', '--version', dest="version", default=None, help="if given, this value will limit the output files to those belonging to a particular version of the database. (defaults to '%(default)s')", choices=db.versions)
  dump_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=SUPPRESS)
  dump_parser.set_defaults(func=dumplist) #action

  # add the checkfiles command
  check_message = "Check if the files exist, based on your criteria"
  check_parser = subparsers.add_parser('checkfiles', help=check_message)
  check_parser.add_argument('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  check_parser.add_argument('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  check_parser.add_argument('-c', '--class', dest="cls", default=None, help="if given, limits the dump to a particular subset of the data that corresponds to the given class (defaults to '%(default)s')", choices=db.classes)
  check_parser.add_argument('-g', '--group', dest="group", default=None, help="if given, this value will limit the output files to those belonging to a particular group. (defaults to '%(default)s')", choices=db.groups)
  check_parser.add_argument('-v', '--version', dest="version", default=None, help="if given, this value will limit the output files to those belonging to a particular version of the database. (defaults to '%(default)s')", choices=db.versions)
  check_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=SUPPRESS)
  check_parser.set_defaults(func=checkfiles) #action

  # add the dumpfold command
  dumpfold_message = "Dumps list of files belonging to a certain cross-fold validation fold based on your criteria"
  dumpfold_parser = subparsers.add_parser('dumpfold', help=dumpfold_message)
  #dumpfold_parser.add_argument('-c', '--class', dest="cls", help="limits the dump to a particular subset of the data that corresponds to the given class (defaults to '%(default)s')", choices=db.classes)
  #dumpfold_parser.add_argument('-v', '--version', dest="version", help="this value will limit the output files to those belonging to a particular version of the database. (defaults to '%(default)s')", choices=db.versions)
  dumpfold_parser.add_argument('cls', help="limits the dump to a particular subset of the data that corresponds to the given class (defaults to '%(default)s')", choices=db.classes)
  dumpfold_parser.add_argument('version', help="this value will limit the output files to those belonging to a particular version of the database. (defaults to '%(default)s')", choices=db.versions)
  dumpfold_parser.add_argument('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  dumpfold_parser.add_argument('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  dumpfold_parser.add_argument('--nf', '--fold_no', dest="fold_no", type=int, default=0, help="the number of fold whose files you would like to dump. (defaults to '%(default)s')")
  dumpfold_parser.add_argument('-s', '--subset', dest="subset", default='', help="specifies whether the dumped files should from the validation or the training subset for that particular fold (defaults to '%(default)s')", choices=('training', 'validation'))
  dumpfold_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=SUPPRESS)
  dumpfold_parser.set_defaults(func=dumpfold) #action
