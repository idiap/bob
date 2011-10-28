#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 28 Oct 09:00:55 2011 

"""Legacy file additions
"""

from libpytorch_io import BinFile, TensorFile

def binfile_getitem(self, i):
  """Returns a array<> object with the expected element type and shape"""
  return getattr(self, '__getitem_%s_%d__' % \
      (self.elementType.name, len(self.shape)))(i)

BinFile.__getitem__ = binfile_getitem
del binfile_getitem

def tensorfile_getitem(self, i):
  """Returns a array<> object with the expected element type and shape"""
  return getattr(self, '__getitem_%s_%d__' % \
      (self.elementType.name, len(self.shape)))(i)

TensorFile.__getitem__ = tensorfile_getitem
del tensorfile_getitem


