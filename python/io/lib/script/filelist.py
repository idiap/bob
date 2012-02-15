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

import os, sys
import fileinput
from optparse import OptionParser

def main():
  usage = "usage: %prog [options] <input_files> "

  parser = OptionParser(usage)
  parser.set_description("Add a prefix and/or suffix to each element of a list.")

  parser.add_option("-d",
                    "--dir",
                    dest="dir",
                    help="Base directory",
                    type="string",
                    default=None)
  parser.add_option("-e",
                    "--ext",
                    dest="ext",
                    help="File extension",
                    type="string",
                    default=None)
  parser.add_option("-c",
                    "--check",
                    dest="check",
                    help="Check that each file exists",
                    action="store_true")


  (options, args) = parser.parse_args()


  for line in fileinput.input(args):
    line = line.strip().strip('\r\n')
    if options.dir != None:
      line = os.path.join(options.dir, line)

    if options.ext != None:
      line = line + options.ext

    if options.check and not os.path.exists(line):
      print >> sys.stderr, "No such file: " + line
      sys.exit(1)
      
    try:
      print line
    except IOError as e:
      if e.errno == 32:
        sys.exit(0)
      else:
        raise
