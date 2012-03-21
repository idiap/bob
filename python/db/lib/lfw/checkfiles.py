#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Checks for installed files.
"""

import os
import sys

# Driver API
# ==========

def checkfiles(args):
  """Checks lists of files based on your criteria"""

  from .query import Database
  db = Database()

  r = db.files(
      directory=args.directory,
      extension=args.extension,
      )

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

def add_command(subparsers):
  """Add specific subcommands that the action "checkfiles" can use"""

  from argparse import SUPPRESS

  parser = subparsers.add_parser('checkfiles', help=checkfiles.__doc__)

  parser.add_argument('-d', '--directory', dest="directory", default='', help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
  parser.add_argument('-e', '--extension', dest="extension", default='', help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
  parser.add_argument('--self-test', dest="selftest", default=False,
      action='store_true', help=SUPPRESS)

  parser.set_defaults(func=checkfiles) #action
