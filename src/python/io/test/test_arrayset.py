#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 03 Aug 2011 12:33:08 CEST 

"""Some tests to arraysets
"""

import os, sys
import unittest
import torch
import numpy

class ArraysetTest(unittest.TestCase):
  """Performs various tests for the Torch::io::Arrayset objects"""

  def test01_extend1d(self):

    # shows how to use the extend() method on arraysets.

    t = torch.io.Arrayset()
    data = numpy.array(range(50), 'float32').reshape(25,2)
    t.extend(data, 0)

    self.assertEqual( len(t), 25 )
    self.assertEqual( t.elementType, torch.io.ElementType.float32 )
    self.assertEqual( t.shape, (2,) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(data[i,:], k) )

    # we can achieve the same effect with lists
    t = torch.io.Arrayset()
    vdata = [data[k,:] for k in range(data.shape[0])]
    t.extend(vdata)

    self.assertEqual( len(t), 25 )
    self.assertEqual( t.elementType, torch.io.ElementType.float32 )
    self.assertEqual( t.shape, (2,) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(vdata[i], k) )

  def test02_extend2d(self):

    # shows how to use the extend() method on arraysets.
    t = torch.io.Arrayset()
    data = numpy.array(range(90), 'float64').reshape(3,10,3)
    t.extend(data, 1)

    self.assertEqual( len(t), 10 )
    self.assertEqual( t.elementType, torch.io.ElementType.float64 )
    self.assertEqual( t.shape, (3,3) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(data[:,i,:], k) )

    # we can achieve the same effect with lists once more
    t = torch.io.Arrayset()
    vdata = [data[:,k,:] for k in range(data.shape[1])]
    t.extend(vdata)

    self.assertEqual( len(t), 10 )
    self.assertEqual( t.elementType, torch.io.ElementType.float64 )
    self.assertEqual( t.shape, (3,3) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(vdata[i], k) )

  def test03_extend3d(self):

    # shows how to use the extend() method on arraysets.

    t = torch.io.Arrayset()
    data = numpy.array(range(540), 'complex128').reshape(3,4,15,3)
    t.extend(data, 2)

    self.assertEqual( len(t), 15 )
    self.assertEqual( t.elementType, torch.io.ElementType.complex128 )
    self.assertEqual( t.shape, (3,4,3) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(data[:,:,i,:], k) )

    # we can achieve the same effect with lists once more
    t = torch.io.Arrayset()
    vdata = [data[:,:,k,:] for k in range(data.shape[2])]
    t.extend(vdata)

    self.assertEqual( len(t), 15 )
    self.assertEqual( t.elementType, torch.io.ElementType.complex128 )
    self.assertEqual( t.shape, (3,4,3) )

    for i, k in enumerate(t):
      self.assertTrue ( numpy.array_equal(vdata[i], k) )

  def test04_canLoadMatlab(self):

    # shows we can load a 2D matlab array and interpret it as a bunch of 1D
    # arrays, correctly

    t = torch.io.Arrayset('matlab_1d.hdf5')
    self.assertEqual( len(t), 512 )
    self.assertEqual( t.shape, (1,) )
    self.assertEqual( t.elementType.name, 'float64' )

    t = torch.io.Arrayset('matlab_2d.hdf5')
    self.assertEqual( len(t), 512 )
    self.assertEqual( t.shape, (2,) )
    self.assertEqual( t.elementType.name, 'float64' )

    # interestingly enough, if you load those files as arrays, you will read
    # the whole data at once:

    t = torch.io.Array('matlab_1d.hdf5')
    self.assertEqual( t.shape, (512,) )
    self.assertEqual( t.elementType.name, 'float64' )

    t = torch.io.Array('matlab_2d.hdf5')
    self.assertEqual( t.shape, (512,2) )
    self.assertEqual( t.elementType.name, 'float64' )

if __name__ == '__main__':
  import gc
  import inspect

  exclude = [
      "function",
      "type",
      "list",
      "dict",
      "tuple",
      "wrapper_descriptor",
      "module",
      "method_descriptor",
      "member_descriptor",
      "instancemethod",
      "builtin_function_or_method",
      "frame",
      "classmethod",
      "classmethod_descriptor",
      "_Environ",
      "MemoryError",
      "_Printer",
      "_Helper",
      "getset_descriptor",
      ]

  def dumpObjects():
    gc.collect()
    oo = gc.get_objects()
    for o in oo:
      if getattr(o, "__class__", None):
        name = o.__class__.__name__
        if name not in exclude:
          try:
            filename = inspect.getabsfile(o.__class__)
          except Exception, e:
            print "Cannot get filename of %s" % o.__class__.__name__
            continue

          print "Object of class:", name, "...",
          print "defined in file:", filename
  
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  os.chdir('data')

  suite = unittest.TestLoader().loadTestsFromTestCase(ArraysetTest)
  unittest.TextTestRunner(verbosity=2).run(suite)
  
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()

  #dumpObjects()
