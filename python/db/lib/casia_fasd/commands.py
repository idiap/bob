#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Sun Mar 25 18:15:55 CEST 2012

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

  r = db.files(directory=args.directory, extension=args.extension, groups=args.group, cls=args.cls, qualities=args.quality, types=args.type)

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  for id, f in r.items():
    output.write('%s\n' % (f,))
  


def checkfiles(args):
  """Checks the existence of the files based on your criteria""" 
    
  from .__init__ import Database
  db = Database()

  r = db.files(directory=args.directory, extension=args.extension, groups=args.group, cls=args.cls, qualities=args.quality, types=args.type)

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
      help="CASIA Face Anti-Spoofing Database",
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
  dump_parser.add_argument('-q', '--quality', dest="quality", default=None, help="if given, this value will limit the output files to those belonging to a particular quality of recording. (defaults to '%(default)s')", choices=db.qualities)
  dump_parser.add_argument('-t', '--type', dest="type", default=None, help="if given, this value will limit the output files to those belonging to a particular type of attack. (defaults to '%(default)s')", choices=db.types)
  dump_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=SUPPRESS)
  dump_parser.set_defaults(func=dumplist) #action

  # add the checkfiles command
  check_message = "Check if the files exist, based on your criteria"
  check_parser = subparsers.add_parser('checkfiles', help=check_message)
  check_parser.add_argument('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  check_parser.add_argument('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  check_parser.add_argument('-c', '--class', dest="cls", default=None, help="if given, limits the dump to a particular subset of the data that corresponds to the given class (defaults to '%(default)s')", choices=db.classes)
  check_parser.add_argument('-g', '--group', dest="group", default=None, help="if given, this value will limit the output files to those belonging to a particular group. (defaults to '%(default)s')", choices=db.groups)
  check_parser.add_argument('-q', '--quality', dest="quality", default=None, help="if given, this value will limit the output files to those belonging to a particular quality of recording. (defaults to '%(default)s')", choices=db.qualities)
  check_parser.add_argument('-t', '--type', dest="type", default=None, help="if given, this value will limit the output files to those belonging to a particular type of attack. (defaults to '%(default)s')", choices=db.types)
  check_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=SUPPRESS)
  check_parser.set_defaults(func=checkfiles) #action
