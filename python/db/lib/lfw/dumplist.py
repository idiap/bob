#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
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
      groups=args.groups,
      purposes=args.purposes
      )

  output = sys.stdout
  if args.selftest:
    from ..utils import null
    output = null()

  for id, f in r.items():
    output.write('%s\n' % (f,))


def dumppairs(args):
  """Dumps lists of pairs of files based on your criteria"""

  from .query import Database
  db = Database()
  
  r = db.pairs(
      directory=args.directory,
      extension=args.extension,
      protocol=args.protocol,
      groups=args.groups,
      classes=args.classes
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
  from .query import Database

  dumplist_parser = subparsers.add_parser('dumplist', help=dumplist.__doc__)

  dumplist_parser.add_argument('-d', '--directory', help="if given, this path will be prepended to every entry returned.")
  dumplist_parser.add_argument('-e', '--extension', help="if given, this extension will be appended to every entry returned.")
  dumplist_parser.add_argument('-p', '--protocol', default='view1', help="specifies the protocol for which the files should be dumped. (defaults to '%(default)s')", choices=Database().m_valid_protocols)
  dumplist_parser.add_argument('-g', '--groups', help="if given, limits the dump to a particular group of the data.", choices=Database().m_valid_groups)
  dumplist_parser.add_argument('-P', '--purposes', help="if given, limits the dump to a particular purpose.", choices=Database().m_valid_purposes)
  dumplist_parser.add_argument('--self-test', dest="selftest", default=False,
      action='store_true', help=SUPPRESS)

  dumplist_parser.set_defaults(func=dumplist) #action

  dumppairs_parser = subparsers.add_parser('dumppairs', help=dumplist.__doc__)

  dumppairs_parser.add_argument('-d', '--directory', help="if given, this path will be prepended to every entry returned.")
  dumppairs_parser.add_argument('-e', '--extension', help="if given, this extension will be appended to every entry returned.")
  dumppairs_parser.add_argument('-p', '--protocol', default='view1', help="specifies the protocol for which the files should be dumped. (defaults to '%(default)s')", choices=Database().m_valid_protocols)
  dumppairs_parser.add_argument('-g', '--groups', help="if given, limits the dump to a particular group of the data.", choices=Database().m_valid_groups)
  dumppairs_parser.add_argument('-c', '--classes', help="if given, limits the dump to a particular class of pairs.", choices=Database().m_valid_classes)
  dumppairs_parser.add_argument('--self-test', dest="selftest", default=False,
      action='store_true', help=SUPPRESS)

  dumppairs_parser.set_defaults(func=dumppairs) #action
