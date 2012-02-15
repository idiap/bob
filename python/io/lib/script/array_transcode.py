#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Jun 22 17:50:08 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

"""This script transcodes a single file containing an array to a possibly
different format.
"""

import os
import sys
import bob

def print_codecs():
  """Prints all installed codecs and the extensions they cover"""

  print
  print "Built-in extension support:"
  print "----------------+" + 60 * '-'
  print " %-14s | %s" % ("Extension", "Description")
  print "----------------+" + 60 * '-'
  for k, v in bob.io.extensions().iteritems():
    print " %-14s | %s" % (k, v)
  print "----------------+" + 60 * '-'

def main():

  if len(sys.argv) != 3:
    print __doc__
    print "usage: %s from-file to-file" % os.path.basename(sys.argv[0])
    print_codecs()
    sys.exit(1)

  bob.io.open(sys.argv[2], 'w').write(bob.io.open(sys.argv[1], 'r').read())
