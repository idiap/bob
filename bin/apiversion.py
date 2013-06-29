#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sat 29 Jun 10:03:11 2013 

import sys
if len(sys.argv) != 2:
  print "You need to pass the Bob version to create an API version"
  print "e.g.: %s 1.2.0b3" % sys.argv[0]
  sys.exit(1)
from distutils.version import StrictVersion
v = StrictVersion(sys.argv[1])
print("0x%02x%02x" % v.version[0:2])
