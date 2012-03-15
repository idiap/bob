.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
.. Wed Mar 14 12:31:35 2012 +0100
.. 
.. Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
.. 
.. This program is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, version 3 of the License.
.. 
.. This program is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
.. 
.. You should have received a copy of the GNU General Public License
.. along with this program.  If not, see <http://www.gnu.org/licenses/>.

**************************
 Multi-dimensional Arrays
**************************


Introduction
============

The fundamental data structure of |project| consists in multi-dimensional
arrays. In signal-processing and machine learning, arrays are indeed a suitable
representation for many different types of digital signals such as images, 
audio data, extracted features, etc. `Python`_ is the working environment
selected for this library. Furtherore, we have decided to rely on existing
`NumPy`_ multi-dimensional arrays.


Numpy basics
============

A `NumPy`_ array is a table of elements, all of the same type, indexed by a 
tuple of positive integers. Before using any of the functionalities described
below, `NumPy`_ should be imported in the `Python`_ environment.

.. doctest::

   >>> import numpy


Array creation
~~~~~~~~~~~~~~

There are different ways to create `NumPy`_ arrays. For instance, to create an
array with initialized content:

.. testsetup:: *

   import numpy

.. doctest::

   >>> A = numpy.array([[1,2,3,4],[5,6,7,8]]) # Creates a 2D array with initialized values
   >>> print A
   [[1 2 3 4]
    [5 6 7 8]]

It is also possible just to allocate the array in memory.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> B = numpy.ndarray((2,4)) # Creates a 2D array with uninitialized values
   >>> B.fill(7) # Fill it in with a constant value 7
   >>> print B
   [[ 7. 7. 7. 7.]
    [ 7. 7. 7. 7.]]

In both previous cases, `NumPy`_ creates an instance of the class **ndarray**,
which is also known by the alias **array**. The most important attributes of 
an **ndarray** object are:

* ndarray.ndim: the number of dimensions of the array.

* ndarray.shape: the dimensions of the array (a tuple of integers indicating the size of the array in each dimension)

* ndarray.dtype: an object describing the type of the elements in the array.


Accessing array elements
~~~~~~~~~~~~~~~~~~~~~~~~

The operator [] allows to index the elements of an array. Please note that 
the indices start at 0.

.. doctest::

   >>> A = numpy.array([[1,2,3,4],[5,6,7,8]]) # Creates a 2D array with initialized values
   >>> print A[0,1]
   2

It is also possible to iterate over an array. In the case of a 
multi-dimensional array, this is done with respect to the first dimension.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> for row in A: print row
   [1 2 3 4]
   [5 6 7 8]


Array type
~~~~~~~~~~

The type of the elements of an array can be specified at the creation time.

.. doctest::

   >>> C = numpy.array( [[1,2], [3,4]], dtype='float64' )
   >>> print C.dtype
   float64


If we would like to cast the elements of an array to another type, `NumPy`_ 
provides the **astype()** function.

.. doctest::

   >>> D = C.astype('uint8')
   >>> print D.dtype
   uint8


Array shape
~~~~~~~~~~~

`NumPy`_ provides several features to reshape or stack arrays, such as the
**reshape()**, **hstack()** and **vstack()** functions.

.. doctest::

   >>> E = D.reshape((1,4))
   >>> print E.shape
   (1, 4)
   >>> a = numpy.array( [1,2], dtype='uint8' )
   >>> b = numpy.array( [3,4], dtype='uint8' )
   >>> F = numpy.vstack((a,b))
   >>> print F
   [[1 2]
    [3 4]]
   >>> G = numpy.hstack((a,b))
   >>> print G
   [1 2 3 4]


Mathematical operations
~~~~~~~~~~~~~~~~~~~~~~~

`NumPy`_ also provides numerous mathematical operations. Most of them are 
performed **elementwise**. For instance,

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> a = numpy.array([1,2,3,4])
   >>> b = numpy.array([4,3,2,1])
   >>> c = a+b
   >>> print c
   [5 5 5 5]
   >>> d = a*b
   >>> print d
   [4 6 6 4]
   >>> e = numpy.exp(a)
   >>> print e
   [ 2.71828183 7.3890561 20.08553692 54.59815003]


`NumPy`_ also provides reduction operations.

.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
   >>> print a.sum()
   10
   >>> print a.max()
   4

Partial reductions along a specific dimension are also possible.

.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
   >>> A = numpy.array([[1,2,3,4],[5,6,7,8]]) # Creates a 2D array with initialized values
   >>> print A.sum(axis=0)
   [ 6 8 10 12]
   >>> print A.max(axis=1)
   [4 8]



Assignment, shallow and deep copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different arrays might share the same data in memory. Let's first have a look
at the assignment operator =.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> a = numpy.array([1,2,3,4], dtype='uint8')
   >>> b = a # Asignment -> No copy at all
   >>> print b is a # a and b are two names for the same ndarray object
   True

Furthermore, the assignment operator only creates an alias to the same 
**ndarray** object. In contrast, the **view()** method creates a new 
`NumPy`_ array object that points to the same memory block. This is known as a
shallow copy.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> c = a.view()
   >>> print c is a  # a and b are two different ndarray objects
   False
   >>> c[2] = 7  # but they share the same data in memory
   >>> print a
   [1 2 7 4]

In a similar way, an ndarray might be slice, and in this case, the data are 
still shared between the two **ndarray** instances.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> d = a[0:3] # d is a slice of a (elements 0 to 3 excluded)
   >>> print d is a  # a and d are two different ndarray objects
   False
   >>> print len(d)
   3
   >>> d[0] = 0 # but they share the same data in memory
   >>> print a
   [0 2 7 4]

If we would like to do a deep copy(), we could use the `NumPy`_ **copy()**
function.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> e = a.copy() # e is a complete copy of a
   >>> e[0] = 7
   >>> print a
   [0 2 7 4]

For a more exhaustive introduction about `NumPy`_, please have a look at its 
`user guide`_. For `matlab`_ users, this `page`_ might also be of 
interest.


Signals as multi-dimensional arrays
===================================

* Images/Videos/Audio sequences as numpy array

.. Place here your external references

.. _python: http://www.python.org
.. _numpy: http://numpy.scipy.org
.. _user guide: http://docs.scipy.org/doc/numpy/user/
.. _matlab: http://www.mathworks.ch/products/matlab/
.. _page: http://www.scipy.org/NumPy_for_Matlab_Users page
