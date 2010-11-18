#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 16 Nov 2010 15:54:24 CET 

"""Various tests for the blitz::Array <-> python interface
"""

import os, sys
import unittest

import torch

# Please note that importing torch should pre-load numpy. This second import
# should just bind the name to this module.
import numpy

class ArrayTest(unittest.TestCase):
  """Performs various tests for the blitz::Array<> object."""
 
  def test01_canTalkWithNumpy(self):
    
    #This test demonstrates how to create torch arrays (based on
    #blitz::Array<>) from numpy arrays and vice-versa.

    # Creates an starting from a numpy array. Please note that the type of
    # array is picked-up automatically by Torch and it depends on the
    # architecture you are running on. In the default case, numpy will create
    # arrays of 32 bit integers for a constructor like it follows, if you are
    # on a 32-bit machine. If you would be in a 64-bit machine, the default for
    # numpy is to create 64-bit integers.
    np_array = numpy.array([1, 2, 3, 4, 5, 6]).reshape((2, 3)) 
    t5_array = torch.core.array.as_torch(np_array) #here, t5_array type:??
    self.assertEqual(isinstance(t5_array, (torch.core.array.int32_2,
      torch.core.array.int64_2)), True)

    # You can check several properties of the newly created torch array like
    # this:
    self.assertEqual(t5_array.dimensions(), 2)

    # You can use the extent() method to find out what is the length of the
    # array along a certain dimension. Remember that the first dimension is 0
    # (zero) and not 1. You can use our blitz aliases to address the dimensions
    # as shown bellow:
    self.assertEqual(t5_array.extent(torch.core.array.firstDim), 2)
    self.assertEqual(t5_array.extent(torch.core.array.secondDim), 3)

    # There are shorcuts for extent(firstDim), extent(secondDim) and
    # extent(thirdDim). They are respectively rows(), columns() and depth().
    self.assertEqual(t5_array.extent(torch.core.array.firstDim),
        t5_array.rows())
    self.assertEqual(t5_array.extent(torch.core.array.secondDim),
        t5_array.columns())
    
    # The size is the total size of the array (all dimensions multiplied)
    self.assertEqual(t5_array.size(), 6)

    # The rank is the same as the matrix number of dimensions
    self.assertEqual(t5_array.rank(), 2)

    # Please note that the number of dimensions of blitz::Array<>'s in python is
    # attached to the type name. "float32_3" is a 32-bit float array with 3
    # dimensions. This is the exact equivalent of the C++ declaration
    # blitz::Array<float, 3>.
    
    # You can force the converted type by using an array constructor.
    t5_float_array = torch.core.array.float32_2(np_array)

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  #os.chdir(os.path.join('data', 'video'))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
