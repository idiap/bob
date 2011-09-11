#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Dumps lists of files.
"""

import os
import sys

# Driver API
# ==========

def dumplist(args):
  """Dumps lists of pairs of files based on your criteria"""

  from .query import Database
  db = Database()

  r = db.pairs(
      directory=args.directory,
      extension=args.extension,
      view=args.view,
      subset=args.subset,
      classes=args.classes,
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

  parser.add_argument('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  parser.add_argument('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  parser.add_argument('-v', '--view', dest="view", default='', help="if given, limits the dump to a particular view of the data. (defaults to '%(default)s')", choices=('view1', 'view2', ''))
  parser.add_argument('-s', '--subset', dest="subset", default='', help="if given, limits the dump to a particular subset of the view/data. (defaults to '%(default)s')", choices=('train', 'test', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10', ''))
  parser.add_argument('-c', '--classes', dest="classes", default='', help="if given, this value will limit the output files to those belonging to the given classes. (defaults to '%(default)s')", choices=('matched', 'unmatched', ''))
  parser.add_argument('--self-test', dest="selftest", default=False,
      action='store_true', help=SUPPRESS)

  parser.set_defaults(func=dumplist) #action
