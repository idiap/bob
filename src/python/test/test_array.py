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

  def test01_canCreateFromScratch(self):

    # This test demonstrates how to create torch arrays (blitz::Array<>'s) in
    # python, from scratch. Each torch supported array type exists in python
    # inside the torch.core.array namespace. The arrays are named according to
    # their element type, followed by a "_" and the number of dimensions. This
    # abstraction provides a 1-to-1 mapping between C++ array types and a
    # pythonic representation. It makes it usable in simple enough terms inside
    # python and requires no effort for 3rd party C++ binding creation.
    # 
    # When you create a torch.core.array.float64_2 you _are_ creating a
    # blitz::Array<double, 2> in C++. If you have torch C++ code bound into
    # python that requires a blitz::Array<double, 2>, just pass a
    # torch.core.array.float64_2 object.
    t5_array = torch.core.array.float64_2([1, 2, 3, 4, 5, 6], (2,3))

    # The shape is a tuple
    self.assertEqual(t5_array.shape(), (2,3))

    # You can also access individual extents like this:
    self.assertEqual(t5_array.extent(torch.core.array.firstDim), 2)
    self.assertEqual(t5_array.extent(torch.core.array.secondDim), 3)

    # Please note that we provide pointers for firstDim (0), secondDim (1),
    # etc, so your code looks more intuitive than: t5_array.extent(1) -- is it
    # the first or the second dimension we are referring to here?

    # You can access each individual value of the array with python indexes, no
    # problem:
    self.assertEqual(t5_array[0,0], 1)
    self.assertEqual(t5_array[1,2], 6)

    # Negative indexing will work à là python
    self.assertEqual(t5_array[0,-1], 3)

    # Indexing an unexisting position throws, like for python containers, an
    # IndexError:
    self.assertRaises(t5_array[0,10], IndexError)

    # TODO: You cannot use slices just yet. The primary reason is that
    # implementing a mixed range/integer indexing support is difficult and
    # would incur in all indexing becoming slower. We will continue to
    # investigate if that could not be done in a simpler way. Keep posted!

    # TODO: For very special purposes, you can create fortran arrays from
    # python. Most array constructors support an extra parameter to specify
    # this. Here is how to do it: ...

  def test02_checkProperties(self):

    # This test demonstrate various properties that are available in all array
    # types.
    t = torch.core.array.uint8_3(range(24), (2,3,4))

    # The extent() is the length of the array in a certain dimension
    self.assertEqual(t.extent(torch.core.array.firstDim), 2)
    self.assertEqual(t.extent(torch.core.array.secondDim), 3)
    self.assertEqual(t.extent(torch.core.array.thirdDim), 4)

    # Please note this will reach the same effect:
    self.assertEqual(t.extent(0), 2)
    self.assertEqual(t.extent(1), 3)
    self.assertEqual(t.extent(2), 4)

    # This shows how many dimensions we have in the array
    self.assertEqual(t.dimensions(), 3)

    # This is exactly the same
    self.assertEqual(t.rank(), 4)

    # These are shortcuts for t.extent(firstDim), t.extent(secondDim) and
    # t.extent(thirdDim)
    self.assertEqual(t.rows(), 2)
    self.assertEqual(t.columns(), 3)
    self.assertEqual(t.depth(), 4)

    # There are no other shortcuts for higher dimensions.

    # The size() operator returns the total number of elements in the array. It
    # is the equivalent of multiplying the extents over all dimensions
    self.assertEqual(t.size(), t.extent(0)*t.extent(1)*t.extent(2))
    self.assertEqual(t.size(), 24) 
    # The call numElements() returns the same as size()
    self.assertEqual(t.numElements(), t.size())

    # The base returns the first valid index of a numpy array. The base is 0
    # for c-style arrays and 1 for fortran-style arrays. There are two versions
    # of the method. One that will return the base for all dimensions as a
    # tuple and a second one that will return the base for a particular
    # dimension you can specify:
    self.assertEqual(t.base(), (0,0,0)) #C-style arrays are created by default
    self.assertEqual(t.base(0), 0)
    self.assertEqual(t.base(1), 0)
    self.assertEqual(t.base(2), 0)

    # The next call tells me if all of the array data is stored continuosly in
    # memory. This is always true if you create yourself the array. If the
    # array is pointing to the data of another array, it may not be the case.
    self.assertTrue(t.isStorageContiguous())
    # TODO: slice and check if storage is not contiguous

    # The next method allows you to access the whole shape of the array in one
    # shot.
    self.assertEqual(t.shape(), (2, 3, 4))

    # This method allows you to access the stride information of the array. You
    # can access it for one dimension or multiple dimensions in one shot. The
    # stride is the distance between the array elements in memory, for that
    # particular dimension. It depends on the array extents and the way that
    # the current array is referencing its memory (owner or borrowed from
    # another array).
    self.assertEqual(t.stride(), (12, 4, 1))
    self.assertEqual(t.stride(0), 12)
    self.assertEqual(t.stride(1), 4)
    self.assertEqual(t.stride(2), 1)

    # The next methods allow you to verify what the automatic converters
    # between C++ and NumPy will do for you
    self.assertEqual(t.numpy_typecode(), 'B')
    self.assertEqual(t.numpy_typename(), 'unsigned char (uint8)')
    self.assertEqual(t.numpy_enum(), torch.core.array.NPY_TYPES.NPY_UBYTE)

  def test03_checkTyping(self):

    # This test demonstrates we support all expected types and these types are
    # correctly expressed as python objects.
    for t in ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
        'uint32', 'uint64', 'float32', 'float64', 'float128', 'complex64', 
        'complex128', 'complex256'):
      for d in (1, 2, 3, 4):
        classname = '%s_%d' % (t, d)
        self.assertTrue(hasattr(torch.core.array, classname))

    # We check that we can instantiate each basic type and that the translation
    # of each type element is the expected python type
    self.assertEqual(type(torch.core.array.bool_1((True,),(1,))[0]), bool)
    self.assertEqual(type(torch.core.array.int8_1((1,),(1,))[0]), int)
    self.assertEqual(type(torch.core.array.int16_1((1,),(1,))[0]), int)
    self.assertEqual(type(torch.core.array.int32_1((1,),(1,))[0]), int)
    self.assertEqual(type(torch.core.array.int64_1((1,),(1,))[0]), long)
    self.assertEqual(type(torch.core.array.uint8_1((1,),(1,))[0]), int)
    self.assertEqual(type(torch.core.array.uint16_1((1,),(1,))[0]), int)
    self.assertEqual(type(torch.core.array.uint32_1((1,),(1,))[0]), int)
    self.assertEqual(type(torch.core.array.uint64_1((1,),(1,))[0]), long)
    self.assertEqual(type(torch.core.array.float32_1((1,),(1,))[0]), float)
    self.assertEqual(type(torch.core.array.float64_1((1,),(1,))[0]), float)
    self.assertEqual(type(torch.core.array.float128_1((1,),(1,))[0]), float)
    self.assertEqual(type(torch.core.array.complex64_1((1,),(1,))[0]), complex)
    self.assertEqual(type(torch.core.array.complex128_1((1,),(1,))[0]), complex)
    self.assertEqual(type(torch.core.array.complex256_1((1,),(1,))[0]), complex)

    # And we check we can set the elements using the specified types
    self.assertIsNone(torch.core.array.bool_1((True,),(1,))[0] = False)

    t = torch.core.array.int8_1((1,),(1,))
    self.assertIsNone(t[0] = 127)
    self.assertIsNone(t[0] = -128)
    self.assertRaises(OverflowError, t.__setitem__, (128,))
    self.assertRaises(OverflowError, t.__setitem__, (-129,))

    t = torch.core.array.int16_1((1,),(1,))
    self.assertIsNone(t[0] = 32767)
    self.assertIsNone(t[0] = -32768)
    self.assertRaises(OverflowError, t.__setitem__, (32768,))
    self.assertRaises(OverflowError, t.__setitem__, (-32769,))

    t = torch.core.array.int32_1((1,),(1,))
    self.assertIsNone(t[0] = 2147483647)
    self.assertIsNone(t[0] = -2147483648)
    self.assertRaises(OverflowError, t.__setitem__, (2147483648,))
    self.assertRaises(OverflowError, t.__setitem__, (-2147483649,))

    t = torch.core.array.int64_1((1,),(1,))
    self.assertIsNone(t[0] = 9223372036854775807L)
    self.assertIsNone(t[0] = -9223372036854775808L)
    self.assertRaises(OverflowError, t.__setitem__, (9223372036854775808L,))
    self.assertRaises(OverflowError, t.__setitem__, (-9223372036854775809L,))

    t = torch.core.array.uint8_1((1,),(1,))
    self.assertIsNone(t[0] = 0)
    self.assertIsNone(t[0] = 255)
    self.assertRaises(OverflowError, t.__setitem__, (-1,))
    self.assertRaises(OverflowError, t.__setitem__, (256,))

    t = torch.core.array.uint16_1((1,),(1,))
    self.assertIsNone(t[0] = 0)
    self.assertIsNone(t[0] = 65535)
    self.assertRaises(OverflowError, t.__setitem__, (-1,))
    self.assertRaises(OverflowError, t.__setitem__, (65536,))

    t = torch.core.array.uint32_1((1,),(1,))
    self.assertIsNone(t[0] = 0)
    self.assertIsNone(t[0] = 4294967296)
    self.assertRaises(OverflowError, t.__setitem__, (-1,))
    self.assertRaises(OverflowError, t.__setitem__, (4294967297,))

    t = torch.core.array.uint64_1((1,),(1,))
    self.assertIsNone(t[0] = 0)
    self.assertIsNone(t[0] = 18446744073709551616L)
    self.assertRaises(OverflowError, t.__setitem__, (-1,))
    self.assertRaises(OverflowError, t.__setitem__, (18446744073709551617L,))

    # TODO: Find how to test float32, 64 and 128 limits and put it here

  def test04_canDoSimpleArithmetics(self):

    # This test demonstrate a few basic operators that are possible with our
    # bindings. The tests can be performed over any type, but we choose the
    # complex type so things are more interesting ;-)

    # BIG WARNING: All operators are applied element-by-element. This package
    # alone does not implement real matrix multiplications.

    start = torch.core.array.complex64_2([complex(0, 1), complex(1, 0),
      complex(1, 0), complex(0, 1)], (2,2))
    constant = complex(2,-2)
    other = start * 2

    # Addition with constant
    result = start + constant
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] + constant)

    # Addition with other array
    result = start + other
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] + other[i,j])

    # Subtraction with constant
    result = start - constant
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] - constant)

    # Subtraction with other array
    result = start - other
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] - other[i,j])

    # Multiplication by constant
    result = start * constant
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] * constant)

    # Multiplication by other array
    result = start * other
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] * other[i,j])

    # Division by constant
    result = start / constant
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] / constant)

    # Division by other array
    result = start / other
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] / other[i,j])

    # Please note you can also do all these in place. Now that I convinced you
    # that all operations result in the expected results, we can just
    # cross-check the in-place operators with the independent ones.
    temp = start.copy()
    temp += constant
    self.assertEqual(temp, start + constant)
    temp = start.copy()
    temp += other
    self.assertEqual(temp, start + other)

    temp = start.copy()
    temp -= constant
    self.assertEqual(temp, start - constant)
    temp = start.copy()
    temp -= other
    self.assertEqual(temp, start - other)
 
    temp = start.copy()
    temp *= constant
    self.assertEqual(temp, start * constant)
    temp = start.copy()
    temp *= other
    self.assertEqual(temp, start * other)
 
    temp = start.copy()
    temp /= constant
    self.assertEqual(temp, start / constant)
    temp = start.copy()
    temp /= other
    self.assertEqual(temp, start / other)

  def test05_canManipulate(self):

    # This test demonstrates some nice manipulations you can do with arrays,
    # all in C++ directly, but bound to python.

    # The transpose() method is a generic transpose algorithm that can swap, in
    # a single operation any dimension from the original array:
    t = torch.core.array.int8_3(range(18), (2, 3, 

  def test06_canTalkWithNumpy(self):
    
    # This test demonstrates how to create torch arrays (based on
    # blitz::Array<>) from numpy arrays and vice-versa.

    # Creates an starting from a numpy array. Please note that the type of
    # array is picked-up automatically by Torch and it depends on the
    # architecture you are running on. In the default case, numpy will create
    # arrays of 32 bit integers for a constructor like it follows, if you are
    # on a 32-bit machine. If you would be in a 64-bit machine, the default for
    # numpy is to create 64-bit integers.
    np_array = numpy.array([1, 2, 3, 4, 5, 6]).reshape((2, 3))
    t5_array = torch.core.array.as_torch(np_array)
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
    
    # You can force the converted type by using an array constructor. Not
    # implemented! Take a look at core/python/array.h to know why.
    #t5_float_array = torch.core.array.float32_2(np_array)

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
