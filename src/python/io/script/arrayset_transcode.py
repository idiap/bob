#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 24 Feb 2011 13:27:32 CET 

"""This script transcodes a single file containing an arrayset to a possibly
different format.
"""

import os
import sys

try:
  import bob
except ImportError, ex:
  print "Python module ImportError: %s" % ex
  print "Tip: have you properly setup your bob environment?"
  sys.exit(2)

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

if len(sys.argv) != 3:
  print __doc__
  print "usage: %s from-file to-file" % os.path.basename(sys.argv[0])
  print_codecs()
  sys.exit(1)

infile = bob.io.open(sys.argv[1], 'r')
outfile = bob.io.open(sys.argv[2], 'w')
for k in range(len(infile)):
  outfile.append(infile.read(k))
