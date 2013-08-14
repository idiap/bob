#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu  6 Sep 09:43:34 2012
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Adds the current build/installation path to scripts installed with bob

This script receives 1 argument: the path to the location where the scripts
are installed.

It assumes it will enter lines in every script like 'bob_*.py', just after
'import sys' which reads:

  sys.path.insert(0, '<path to local site-packages>')
"""

import os
import sys
import fnmatch

CYGWIN=(__import__('platform').system().find('CYGWIN') != -1)

if len(sys.argv) > 2:
  print("Not tunning scripts (DESTDIR environment set to '%s')" % sys.argv[2])
  sys.exit(0)

for script in fnmatch.filter(os.listdir(sys.argv[1]), 'bob_*.py'):

  path = os.path.join(sys.argv[1], script)

  print("Tunning %s script at %s" % (os.path.basename(path), os.path.dirname(path)))
  lib = os.path.abspath(os.path.join( os.path.dirname(sys.argv[1]), 'lib'))
  site = os.path.abspath(os.path.join(
    os.path.dirname(sys.argv[1]),
    'lib',
    'python%d.%d' % sys.version_info[:2],
    'site-packages',
    ))
  out = []
  for l in open(path, 'rt'):
    if l.strip() == 'import sys':
      out.append('\n### start modifications by Bob ###\n')
      if CYGWIN:
        out.append("import os\n")
        out.append("os.environ['PATH'] = os.pathsep.join(['%s', os.environ['PATH']])\n" % lib)
      out.append(l)
      out.append("sys.path.insert(0, '%s')\n" % site)
      out.append('### end modifications by Bob ###\n\n')
      continue
    out.append(l)

  # re-write the file - don't leave traces
  open(path, 'wt').writelines(out)
