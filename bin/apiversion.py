#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sat 29 Jun 10:03:11 2013 
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

import sys
if len(sys.argv) != 2:
  print("You need to pass the Bob version to create an API version")
  print("e.g.: %s 1.2.0b3" % sys.argv[0])
  sys.exit(1)
from distutils.version import StrictVersion
v = StrictVersion(sys.argv[1])
print("0x%02x%02x" % v.version[0:2])
