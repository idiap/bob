#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sat 29 Jun 10:03:11 2013 
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

import sys
if len(sys.argv) != 2:
  print("You need to pass the Bob version to create an library version")
  print("e.g.: %s 1.2.0b3" % sys.argv[0])
  sys.exit(1)
from distutils.version import StrictVersion
v = StrictVersion(sys.argv[1])
print("%d.%d" % v.version[0:2])
