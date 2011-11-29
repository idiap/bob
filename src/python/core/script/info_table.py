#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 27 Apr 22:59:28 2011 

"""Prints the version of Torch and exits.
"""

import os
import sys
import torch
import numpy

def version_table():
  """Returns a summarized version table of all software compiled in, with their
  respective versions."""

  space = ' '
  packsize = 20
  descsize = 55 

  version_dict = {}
  version_dict.update(torch.core.version)
  version_dict.update(torch.io.version)
  version_dict.update(torch.sp.version)
  version_dict.update(torch.ip.version)

  try: #the visioner may be compiled in, or not
    version_dict.update(torch.visioner.version)
  except AttributeError:
    pass

  torch_version = "%(TORCH_VERSION)s (%(TORCH_PLATFORM)s)" % os.environ
  print 75*'='
  print (" Torch5spro version %s" % torch_version).center(75)
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

  for k in sorted(torch.io.extensions().keys()):
    v = torch.io.extensions()[k]
    print fmt % (k.ljust(packsize), v.ljust(descsize))
  print sep

version_table()
print_codecs()
sys.exit(0)
