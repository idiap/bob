#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Apr 27 23:16:03 2011 +0200
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

"""Prints the version of bob and exits.
"""

import os
import sys
import bob
import numpy
from ... import build

def version_table():
  """Returns a summarized version table of all software compiled in, with their
  respective versions."""

  space = ' '
  packsize = 20
  descsize = 55 

  version_dict = {}
  version_dict.update(bob.core.version)
  version_dict.update(bob.io.version)
  version_dict.update(bob.sp.version)
  version_dict.update(bob.ip.version)
  if hasattr(bob.machine, 'version'): 
    version_dict.update(bob.machine.version)
  version_dict.update(bob.daq.version)

  try: #the visioner may be compiled in, or not
    version_dict.update(bob.visioner.version)
  except AttributeError:
    pass

  bob_version = "%s (%s)" % (build.version, build.platform)
  print 75*'='
  print (" bob version %s" % bob_version).center(75)
  print 75*'='
  print ""

  print "Built-in Software"
  print "-----------------\n"

  sep = space + packsize*'=' + space + descsize*'='
  fmt = 2*space + ('%%%ds' % packsize) + space + ('%%%ds' % descsize)
  print sep
  print fmt % ('Package'.ljust(packsize), 'Version'.ljust(descsize))
  print sep
  for k in sorted(version_dict.keys()):
    v = version_dict[k]
    if k.lower() == 'numpy': v = '%s (%s)' % (numpy.version.version, v)
    if k.lower() == 'compiler': v = '-'.join(v)
    elif k.lower() == 'ffmpeg':
      if v.has_key('ffmpeg'): v = v['ffmpeg']
      else: v = ';'.join(['%s-%s' % (x, v[x]) for x in v.keys()])
    elif k.lower() == 'qt4': v = '%s (from %s)' % v
    elif k.lower() == 'fftw': v = '%s (%s)' % v[:2]
    print fmt % (k.ljust(packsize), v.ljust(descsize))
  print sep

def print_codecs():
  """Prints all installed codecs and the extensions they cover"""
 
  print ""
  print "Available Codecs"
  print "----------------\n"

  space = ' '
  packsize = 20
  descsize = 55 

  sep = space + packsize*'=' + space + descsize*'='
  fmt = 2*space + ('%%%ds' % packsize) + space + ('%%%ds' % descsize)
  print sep
  print fmt % ('Extension'.ljust(packsize), 'Description'.ljust(descsize))
  print sep

  for k in sorted(bob.io.extensions().keys()):
    v = bob.io.extensions()[k]
    print fmt % (k.ljust(packsize), v.ljust(descsize))
  print sep

def main():
  version_table()
  print_codecs()
  sys.exit(0)
