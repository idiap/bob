#!/usr/bin/env python

import os, sys
import fileinput
from optparse import OptionParser

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