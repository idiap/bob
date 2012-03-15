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

A `NumPy`_ array is a table of elements, all of the same type, indexed by a tuple 
of positive integers. Before using any of the functionalities described below, 
`NumPy`_ should be imported in the `Python`_ environment.

.. doctest::

   >>> import numpy


Array creation
~~~~~~~~~~~~~~

There are different ways to create `NumPy`_ arrays. For instance,
to create an array with initialized content:

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

In both previous cases, `NumPy`_ creates an instance of the class **ndarray**, which
is also known by the alias **array**. The most important attributes of an **ndarray**
object are:

* ndarray.ndim: the number of dimensions of the array.

* ndarray.shape: the dimensions of the array (a tuple of integers indicating the size of the array in each dimension)

* ndarray.dtype: an object describing the type of the elements in the array.


Array indexing
~~~~~~~~~~~~~~

The operator [] allows to index the elements of the array. Please
note that the first index is 0.

.. doctest::

   >>> print B[0,1]
   7.0

It is also possible to iterate over an array. In the case of a 
multi-dimensional array, this is done with respect to the first dimension.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> for row in B: print row
   [ 7. 7. 7. 7.]
   [ 7. 7. 7. 7.]


Array type
~~~~~~~~~~

The type of the elements of an array can be specified at the creation time.

.. doctest::

   >>> C = numpy.array( [[1,2], [3,4]], dtype='float64' )
   >>> print C.dtype
   float64


If we would like to cast the elements of an array to another type, 
`NumPy`_ provides the **astype()** function.

.. doctest::

   >>> D = C.astype('uint8')
   >>> print D.dtype
   uint8


Mathematical operations
~~~~~~~~~~~~~~~~~~~~~~~

`NumPy`_ provides numerous mathematical operations. Most of them are performed
**elementwise**. For instance,

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
   12
   >>> print a.max()
   4


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

In contrast, the **view()** method creates a new `NumPy`_ array object that 
points to the same memory block. This is known as a shallow copy.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> c = a.view()
   >>> print c is a  # a and b are two different ndarray objects
   False
   >>> c[2] = 7  # but they share the same data in memory
   >>> print a
   [1 2 7 4]

In a similar way, an ndarray might be slice, and in this case, the data are 
still shared


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

For a more exhaustive introduction, please consider the ...

Signals as multi-dimensional arrays
===================================

* Images/Videos/Audio sequences as numpy array

.. Place here your external references

.. _python: http://www.python.org
.. _numpy: http://numpy.scipy.org
