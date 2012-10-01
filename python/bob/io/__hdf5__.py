#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 28 Oct 08:59:10 2011 

"""HDF5 additions
"""

from ._io import HDF5Type, HDF5File

# Some HDF5 addons
def hdf5type_str(self):
  return "%s@%s" % (self.type_str(), self.shape())
HDF5Type.__str__ = hdf5type_str
del hdf5type_str

def hdf5type_repr(self):
  return "<HDF5Type: %s (0x%x)>" % (str(self), id(self))
HDF5Type.__repr__ = hdf5type_repr
del hdf5type_repr
