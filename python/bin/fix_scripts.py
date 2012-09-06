#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu  6 Sep 09:43:34 2012 

"""Adds the current build/installation path to scripts installed with bob

This script receives 1 argument: the path to the location where the scripts 
are installed. 

It assumes it will enter a single line in every script like 'bob_*.py', just
after 'import sys' which reads: 

  sys.path.insert(0, '<path to local site-packages>')
"""

import os
import sys
import fnmatch

for script in fnmatch.filter(os.listdir(sys.argv[1]), 'bob_*.py'):

  path = os.path.join(sys.argv[1], script)
  print "Tunning %s script at %s" % (os.path.basename(path), os.path.dirname(path))
  site = os.path.abspath(os.path.join(
    os.path.dirname(sys.argv[1]),
    'lib', 
    'python%d.%d' % sys.version_info[:2],
    'site-packages',
    ))
  out = []
  for l in open(path, 'rt'):
    if l.strip() == 'import sys':
      out.append(l)
      out.append("sys.path.insert(0, '%s')\n" % site)
      continue
    out.append(l)

  # re-write the file - don't leave traces
  open(path, 'wt').writelines(out)
