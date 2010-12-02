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
    self.assertRaises(IndexError, t5_array.__getitem__, (0,10))

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
    self.assertEqual(t.rank(), 3)

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
    t = torch.core.array.bool_1((True,),(1,))
    t[0] = True
    self.assertEqual(t[0], True)

    t = torch.core.array.int8_1((1,),(1,))
    t[0] = 127
    self.assertEqual(t[0], 127)
    t[0] = -128
    self.assertEqual(t[0], -128)
    self.assertRaises(OverflowError, t.__setitem__, 0, 128)
    self.assertRaises(OverflowError, t.__setitem__, 0, -129)

    t = torch.core.array.int16_1((1,),(1,))
    t[0] = 32767
    self.assertEqual(t[0], 32767)
    t[0] = -32768
    self.assertEqual(t[0], -32768)
    self.assertRaises(OverflowError, t.__setitem__, 0, 32768)
    self.assertRaises(OverflowError, t.__setitem__, 0, -32769)

    t = torch.core.array.int32_1((1,),(1,))
    t[0] = 2147483647
    self.assertEqual(t[0], 2147483647)
    t[0] = -2147483648 
    self.assertEqual(t[0], -2147483648)
    self.assertRaises(OverflowError, t.__setitem__, 0, 2147483648)
    self.assertRaises(OverflowError, t.__setitem__, 0, -2147483649)

    t = torch.core.array.int64_1((1,),(1,))
    t[0] = 9223372036854775807L
    self.assertEqual(t[0], 9223372036854775807L)
    t[0] = -9223372036854775808L
    self.assertEqual(t[0], -9223372036854775808L)
    self.assertRaises(OverflowError, t.__setitem__, 0,9223372036854775808L)
    self.assertRaises(OverflowError, t.__setitem__, 0,-9223372036854775809L)

    t = torch.core.array.uint8_1((1,),(1,))
    t[0] = 0
    self.assertEqual(t[0], 0)
    t[0] = 255
    self.assertEqual(t[0], 255)
    # The following will work with a positive underflow. This is why: the
    # assigned number (-1) will be converted to a 32-bit positive number which
    # is 4294967296 and that cannot be assigned to an 8-bit value.
    #self.assertRaises(OverflowError, t.__setitem__, 0,-1)
    self.assertRaises(OverflowError, t.__setitem__, 0,256)

    t = torch.core.array.uint16_1((1,),(1,))
    t[0] = 0
    self.assertEqual(t[0], 0)
    t[0] = 65535
    self.assertEqual(t[0], 65535)
    # The same affirmation from the previous test applies.
    #self.assertRaises(OverflowError, t.__setitem__, 0,-1)
    self.assertRaises(OverflowError, t.__setitem__, 0,65536)

    t = torch.core.array.uint32_1((1,),(1,))
    t[0] = 0
    self.assertEqual(t[0], 0)
    t[0] = 4294967295
    self.assertEqual(t[0], 4294967295)
    # Please note that this will not raise as expected... The reason from the
    # previous tests apply.
    #self.assertRaises(OverflowError, t.__setitem__, 0,-1)
    self.assertRaises(OverflowError, t.__setitem__, 0,4294967296)

    t = torch.core.array.uint64_1((1,),(1,))
    t[0] = 0
    self.assertEqual(t[0], 0)
    t[0] = 18446744073709551615L 
    self.assertEqual(t[0], 18446744073709551615L)
    # Same restriction here.
    #self.assertRaises(OverflowError, t.__setitem__, 0,-1)
    self.assertRaises(OverflowError, t.__setitem__, 0,18446744073709551616L)

    # TODO: Find how to test float32, 64 and 128 limits and put it here

  def test04_canDoSimpleArithmetics(self):

    # This test demonstrate a few basic operators that are possible with our
    # bindings. The tests can be performed over any type, but we choose the
    # complex type so things are more interesting ;-)

    # BIG WARNING: All operators are applied element-by-element. This package
    # alone does not implement real matrix multiplications.

    start = torch.core.array.complex128_2([complex(0, 1), complex(1, 0),
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

    # The power function can be applied to floats and complex, but not to
    # integers. If you need to apply it to integers, convert back and forth or
    # just use numpy directly and cast the result back to a torch array.
    result = start ** other
    for i in range(2):
      for j in range(2):
        diff = result[i,j] - ( start[i,j] ** other[i,j] )
        self.assertTrue(abs(diff) < 1e-10)

    # You can do the powering with constants as well
    result = start ** constant
    for i in range(2):
      for j in range(2):
        diff = result[i,j] - ( start[i,j] ** constant )
        self.assertTrue(abs(diff) < 1e-10)

    # Reminder of the division by constant. Not applicable to float and
    # complex arrays.
    start = torch.core.array.int16_2(range(9), (3,3))
    constant = 2

    result = start % constant
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] % constant)
    
    # Reminder of the division by array.
    other = start.copy()
    other.fill(2) # an array filled with 2's
    result = start % other
    for i in range(2):
      for j in range(2):
        self.assertEqual(result[i,j], start[i,j] % 2)

    # Please note you can also do all these in place. Now that I convinced you
    # that all operations result in the expected results, we can just
    # cross-check the in-place operators with the independent ones.
    temp = start.copy()
    temp += constant
    self.assertTrue(temp, start + constant)
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

    # TODO: Tests for &, |, ^, <<, >>

  def test05_canDoBooleanOperations(self):

    # This will test all sorts of boolean operations on integers. These
    # operations are also available for floats and complex numbers. Care must
    # be taken when using floats and complex numbers as equality is a
    # tricky thing. Please note that operators __lt__ and the such are not
    # available for complex numbers for obvious reasons.

    # The return type of a comparison operator is always a boolean array, with
    # the number of dimensions and extents equal to both arrays. It is illegal
    # and unsupported to compare arrays with a different number of dimensions. 
    # This brings us to another concern: what if the arrays have different
    # extents? Well, in this case blitz++ will compare just the extents that
    # have equal length.

    a1 = torch.core.array.float32_2(range(6), (2,3))
    a2 = torch.core.array.float32_2(range(6), (2,3))
    cmp = (a1 == a2)

    self.assertTrue(isinstance(cmp, torch.core.array.bool_2))
    self.assertEqual(cmp.shape(), a1.shape())
    self.assertEqual(cmp.shape(), a2.shape())
    self.assertTrue(cmp.all())

    a3 = torch.core.array.float32_2(range(12), (4,3))
    cmp = (a1 == a3)
    self.assertTrue(isinstance(cmp, torch.core.array.bool_2))
    self.assertEqual(cmp.shape(), (1,3))
    self.assertTrue(cmp.all()) #because a3 and a1 have equal (0,1),(0,2),(0,3)
    # if you are curious - yes, looks like it is a blitz++ limitation.

    # Other operators
    a2 += 1 #make a2 bigger than a1, element-wise
    
    cmp = (a1 != a2)
    self.assertTrue(isinstance(cmp, torch.core.array.bool_2))
    self.assertEqual(cmp.shape(), a1.shape())
    self.assertEqual(cmp.shape(), a2.shape())
    self.assertTrue(cmp.all())

    cmp = (a1 < a2)
    self.assertTrue(isinstance(cmp, torch.core.array.bool_2))
    self.assertEqual(cmp.shape(), a1.shape())
    self.assertEqual(cmp.shape(), a2.shape())
    self.assertTrue(cmp.all())

    cmp = (a1 <= a2)
    self.assertTrue(isinstance(cmp, torch.core.array.bool_2))
    self.assertEqual(cmp.shape(), a1.shape())
    self.assertEqual(cmp.shape(), a2.shape())
    self.assertTrue(cmp.all())

    cmp = (a2 > a1)
    self.assertTrue(isinstance(cmp, torch.core.array.bool_2))
    self.assertEqual(cmp.shape(), a1.shape())
    self.assertEqual(cmp.shape(), a2.shape())
    self.assertTrue(cmp.all())

    cmp = (a2 >= a1)
    self.assertTrue(isinstance(cmp, torch.core.array.bool_2))
    self.assertEqual(cmp.shape(), a1.shape())
    self.assertEqual(cmp.shape(), a2.shape())
    self.assertTrue(cmp.all())

  def test06_canDoMath(self):

    # This test demonstrates how to apply some math to the arrays. A few
    # methods from blitz++ were bound to each of the array types available. Not
    # all methods can be called from every array type just because it does not
    # make sense all the time. E.g., you cannot call most of these methods with
    # boolean arrays.

    import math # not necessary, just to cross-check the results from blitz++
    import numpy # not necessary, just to cross-check the results from blitz++

    def test_math_op_2dim(array, result, python_equivalent):
      """A generic test method so we don't have to be repetitive. All
      mathematical operators are applied by calling it directly on the array
      E.g.: array.exp()."""
      for i in range(result.extent(torch.core.array.firstDim)):
        for j in range(result.extent(torch.core.array.secondDim)):
          self.assertEqual(result[i,j], python_equivalent(array[i,j]))

    # First some methods applicable to integers, unsigned integers and floats.
    # Please note that, to compare the blitz++ operators and python math
    # operators we should always use double precision as that is the default
    # inside the python "math" module implementation. You can still use
    # everything with single precision. Results will be similar.
    t = torch.core.array.float64_2([0.1, 0.2, 0.3, 0.4], (2,2))

    test_math_op_2dim(t, t.acos(), math.acos)
    test_math_op_2dim(t, t.asin(), math.asin)
    test_math_op_2dim(t, t.atan(), numpy.arctan)
    test_math_op_2dim(t, t.cos(), math.cos)
    test_math_op_2dim(t, t.cosh(), numpy.cosh)
    # we need slightly bigger numbers to apply acosh, of course.
    test_math_op_2dim(t*10, (t*10).acosh(), numpy.arccosh)
    test_math_op_2dim(t, t.log(), math.log)
    test_math_op_2dim(t, t.log10(), math.log10)
    test_math_op_2dim(t, t.sin(), math.sin)
    test_math_op_2dim(t, t.sinh(), numpy.sinh)
    test_math_op_2dim(t, t.sqr(), numpy.square)
    test_math_op_2dim(t, t.sqrt(), math.sqrt)
    test_math_op_2dim(t, t.tan(), math.tan)
    test_math_op_2dim(t, t.tanh(), numpy.tanh)
    test_math_op_2dim(t, t.atanh(), numpy.arctanh)
    #test_math_op_2dim(t, t.cbrt(), ??)
    test_math_op_2dim(t, t.exp(), math.exp)
    test_math_op_2dim(t, t.expm1(), numpy.expm1)
    #test_math_op_2dim(t, t.erf(), ??)
    #test_math_op_2dim(t, t.erfc(), ??)
    #test_math_op_2dim(t, t.ilogb(), ??)
    #test_math_op_2dim(t, t.j0(), ??)
    #test_math_op_2dim(t, t.j1(), ??)
    #test_math_op_2dim(t, t.lgamma(), ??)
    test_math_op_2dim(t, t.log1p(), numpy.log1p)
    test_math_op_2dim(t, t.rint(), numpy.rint)
    #test_math_op_2dim(t, t.y0(), ??)
    #test_math_op_2dim(t, t.y1(), ??)
    
    # Please note that python provides a "__abs__" slot for calls like abs(x).
    # We keep the pythonic approach, so the call is a little bit different that
    # in other cases, but will work by calling the bound C++ method.
    test_math_op_2dim(t, abs(t), numpy.abs)

    # Now some methods that applicable to float arrays only
    test_math_op_2dim(t, t.ceil(), math.ceil)
    test_math_op_2dim(t, t.floor(), math.floor)

    # And some operators that we can apply on complex only
    t = torch.core.array.complex64_2([complex(0, 1), complex(1, 0),
      complex(1, 0), complex(0, 1)], (2,2))
    #test_math_op_2dim(t, t.arg(), ??)
    test_math_op_2dim(t, t.conj(), numpy.conjugate)

    # And to make sure the abs of a complex is the vector size.
    test_math_op_2dim(t, abs(t), numpy.abs)

  def test07_canDoReductions(self):

    # This test demonstrates how to reduce the array into a single element.
    # Partial reductions are still not implemented.
    a = torch.core.array.float32_2(range(4), (2, 2))

    self.assertTrue(isinstance(a.sum(), float))
    self.assertTrue(a.sum(), sum(range(4)))

    self.assertEqual(a.product(), 0)
    self.assertEqual(a.mean(), sum(range(4))/4.0)
    self.assertEqual(a.min(), 0)
    self.assertEqual(a.max(), 3)
    self.assertEqual(a.minIndex(), (0,0))
    self.assertEqual(a.maxIndex(), (1,1))
    self.assertEqual(a.count(), 3) #count() == where bool(value) is True
    self.assertEqual(a.any(), True) #is any value True?
    self.assertEqual(a.all(), False) #are all values True?

  def test08_canManipulate(self):

    # This test demonstrates some nice manipulations you can do with arrays,
    # all in C++ directly, but bound to python.

    # The transpose() method is a generic transpose algorithm that can swap, in
    # a single operation any dimension from the original array. Please note
    # this does not copy the data at all, just sets another view for it. It is,
    # therefore, very cheap.
    t = torch.core.array.int8_3(range(24), (2, 3, 4))
    t2 = t.transpose(torch.core.array.thirdDim, torch.core.array.firstDim,
        torch.core.array.secondDim)
    self.assertEqual(t2.shape(), (4,2,3))
    for i in range(2):
      for j in range(3):
        for k in range(4):
          self.assertEqual(t[i,j,k], t2[k,i,j])

    # You can also do the transposition in place:
    t2.transposeSelf(torch.core.array.secondDim, torch.core.array.thirdDim,
        torch.core.array.firstDim)
    self.assertTrue((t == t2).all())

    # The second operation is reversing a certain dimension of the array. You
    # can reverse the contents of the array in a single dimension at a time.
    t3 = t.reverse(torch.core.array.firstDim)
    for i in range(2):
      reversed_i = t.extent(torch.core.array.firstDim) - i - 1
      for j in range(3):
        for k in range(4):
          self.assertEqual(t[i,j,k], t3[reversed_i,j,k])

    # Or reverse in place
    t3.reverseSelf(torch.core.array.firstDim)
    self.assertTrue((t==t3).all())

  def test09_canDoUnaryOperators(self):

    # This test demonstrates some unary operations on arrays. The first one is
    # the "-" negative operator that just inverts the sign of every element
    # inside the array. Defined for all except complex arrays for obvious
    # reasons.
    a = torch.core.array.int8_3(range(24), (2, 3, 4))
    a[0,1,2] = -34 #just to spice things up
    b = -a
    self.assertEqual(b[0,1,2], 34)
    self.assertTrue( (a == -b).all() )

    # The second one is the "invert" operator or "~". It just flips all bits of
    # every element. Defined for boolean, integer and unsigned integer arrays.
    # Please note that on integer arrays ~value = (-value) -1
    b = ~a
    self.assertEqual(b[0,1,2], 33)
    self.assertTrue( (a == ~b).all() )

    # Where as in *unsigned integer* arrays, negating every bit will get you to
    # the other axis side ;-). E.g. ~0 == 255 (for 8 bits)
    a = torch.core.array.uint8_3(range(24), (2, 3, 4))
    b = ~a
    self.assertEqual(b[0,0,0], 255)
    self.assertTrue( (a == ~b).all() )

  def test10_canTalkWithNumpy(self):
    
    # This test demonstrates how to create torch arrays (based on
    # blitz::Array<>) from numpy arrays and vice-versa.

    # Creates an starting from a numpy array. Please note that the type of
    # array is picked-up automatically by Torch and it depends on the
    # architecture you are running on. In the default case, numpy will create
    # arrays of 32 bit integers for a constructor like it follows, if you are
    # on a 32-bit machine. If you would be in a 64-bit machine, the default for
    # numpy is to create 64-bit integers.
    np_array = numpy.array([1, 2, 3, 4, 5, 6]).reshape((2, 3))
    t5_array = torch.core.array.int16_2(np_array)

    # Some basic checks
    self.assertEqual(t5_array.dimensions(), 2)
    self.assertEqual(t5_array.extent(torch.core.array.firstDim), 2)
    self.assertEqual(t5_array.extent(torch.core.array.secondDim), 3)
    self.assertEqual(t5_array.size(), 6)
    for i in range(t5_array.extent(torch.core.array.firstDim)):
      for j in range(t5_array.extent(torch.core.array.secondDim)):
        self.assertEqual(np_array[i,j], t5_array[i,j]) #despite the cast!

    # Please note that the number of dimensions of blitz::Array<>'s in python is
    # attached to the type name. "float32_3" is a 32-bit float array with 3
    # dimensions. This is the exact equivalent of the C++ declaration
    # blitz::Array<float, 3>.

    # And we can convert back using the "as_ndarray()" call available in every
    # torch array bound to python. In this example, converting back should just
    # give us the exact same array as before.
    np_array_2 = t5_array.as_ndarray()

    # Some basic checks

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
