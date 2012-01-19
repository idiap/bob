#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 12 May 14:02:28 2011 

"""Dumps lists of files.
"""

import os
import sys

# Driver API
# ==========

def dumplist(args):
  """Dumps lists of files based on your criteria"""

  from .query import Database
  db = Database()

  r = db.files(
      directory=args.directory, 
      extension=args.extension,
      protocol=args.protocol, 
      support=args.support, 
      groups=args.group, 
      cls=args.cls,
      light=args.light,
      )

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  for id, f in r.items():
    output.write('%s\n' % (f,))

def add_command(subparsers):
  """Add specific subcommands that the action "dumplist" can use"""

  from argparse import SUPPRESS

  parser = subparsers.add_parser('dumplist', help=dumplist.__doc__)

  from .query import Database
  from ..utils import location
  from . import dbname

  db = Database()

  if not db.is_valid():
    protocols = ('waiting','for','database','creation')
  else:
    protocols = db.protocols()

  parser.add_argument('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  parser.add_argument('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  parser.add_argument('-c', '--class', dest="cls", default='', help="if given, limits the dump to a particular subset of the data that corresponds to the given class (defaults to '%(default)s')", choices=('real', 'attack', 'enroll'))
  parser.add_argument('-g', '--group', dest="group", default='', help="if given, this value will limit the output files to those belonging to a particular protocolar group. (defaults to '%(default)s')", choices=db.groups())
  parser.add_argument('-s', '--support', dest="support", default='', help="if given, this value will limit the output files to those using this type of attack support. (defaults to '%(default)s')", choices=db.attack_supports())
  parser.add_argument('-x', '--protocol', dest="protocol", default='', help="if given, this value will limit the output files to those for a given protocol. (defaults to '%(default)s')", choices=protocols)
  parser.add_argument('-l', '--light', dest="light", default='', help="if given, this value will limit the output files to those shot under a given lighting. (defaults to '%(default)s')", choices=db.lights())
  parser.add_argument('--self-test', dest="selftest", default=False,
      action='store_true', help=SUPPRESS)

  parser.set_defaults(func=dumplist) #action
