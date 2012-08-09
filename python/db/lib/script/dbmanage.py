#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue Jun 28 17:12:28 2011 +0200
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

"""This script drives all commands from the specific database subdrivers.
"""

epilog = """  For a list of available databases:
  >>> %(prog)s --help

  For a list of actions on a database:
  >>> %(prog)s <database-name> --help
"""

from bob.db.manage import *

def main():

  from argparse import RawDescriptionHelpFormatter
  parser = create_parser(description=__doc__, epilog=epilog,
      formatter_class=RawDescriptionHelpFormatter)
  args = parser.parse_args()
  args.func(args)
