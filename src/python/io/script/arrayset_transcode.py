#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 24 Feb 2011 13:27:32 CET 

"""This script transcodes a single file containing an arrayset to a possibly
different format.
"""

import os, sys

try:
  import torch
except ImportError, ex:
  print "Python module ImportError: %s" % ex
  print "Tip: have you properly setup your torch environment?"
  sys.exit(2)

def print_codecs():
  """Prints all installed codecs and the extensions they cover"""
  print
  print "Torch built-in codecs:"
  print " %-30s | %s" % ("Codecname", "Extensions supported")
  print "--------------------------------+------------------------------"
  for k in torch.io.ArraysetCodecRegistry.getCodecNames():
    codec = torch.io.ArraysetCodecRegistry.getCodecByName(k)
    print " %-30s | %s" % (codec.name(), ", ".join(codec.extensions()))

if len(sys.argv) != 3:
  print __doc__
  print "usage: %s from-file to-file" % os.path.basename(sys.argv[0])
  print_codecs()
  sys.exit(1)

torch.io.arrayset_transcode(sys.argv[1], sys.argv[2])
