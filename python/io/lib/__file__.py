#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 28 Oct 09:51:24 2011 

"""Addons to files.
"""

from libpybob_io import File, typeinfo

def typeinfo_str(self):
  return "%s@%s" % (self.dtype, self.shape)
typeinfo.__str__ = typeinfo_str
del typeinfo_str

def typeinfo_repr(self):
  return "<typeinfo: %s (0x%x)>" % (str(self), id(self))
typeinfo.__repr__ = typeinfo_repr
del typeinfo_repr

def file_iter(self):
  """Allows Files to be iterated in native python"""
  n = 0
  while n != len(self):
    yield self[n]
    n += 1
  raise StopIteration
File.__iter__ = file_iter
del file_iter
