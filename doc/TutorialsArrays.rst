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

The fundamental data structure of |project| is a multi-dimensional
array. In signal-processing and machine learning, arrays are a suitable
representation for many different types of digital signals such as images, 
audio data and extracted features. Python_ is the working environment
selected for this library and so when using Python_ we have relied on the 
existing NumPy_ multi-dimensional arrays (:py:class:`numpy.ndarray`). This
provides with greater flexibility within the Python environment.

At the C++ level, the `Blitz++`_ library is used to handle arrays. Although we
initially bound `Blitz++`_ Arrays in Python, we quickly realized that it
might be more clever to use the existing NumPy_ ndarrays from Python, as they can
directly be processed by numerous existing Python libraries such as NumPy_ and
SciPy_. 

This means that |project| multi-dimensional arrays are represented in Python by
NumPy_ ndarrays. This also implies that there are internal conversion routines
to convert NumPy_ ndarrays from/to `Blitz++`_. As they are done implicitly, the
user has no need to care about this aspect and should just use NumPy_ ndarrays
everywhere.

For an introduction and tutorials about NumPy_ ndarrays, just 
visit the Numpy_ website.

NumPy_ basics
=============

A NumPy_ array is a table of elements, all of the same type, indexed by a 
tuple of positive integers. Before using any of the functionalities described
below, NumPy_ should be imported in the Python_ environment.

.. doctest::

  >>> import numpy

.. note::

   For `MATLAB`_ users, this `page`_ highlights the differences and the commonalities between `NumPy`_ and `MATLAB`_.

Array creation
--------------

There are different ways to create NumPy_ arrays. For instance, to create an
array with initialized content:

.. testsetup:: *

  import numpy, bob

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

In both previous cases, NumPy_ creates an instance of the class :py:class:`numpy.ndarray`,
which is also known by the alias :py:func:`numpy.array`. The most important attributes of 
an :py:class:`numpy.ndarray` object are:

* :py:attr:`numpy.ndarray.ndim`: the number of dimensions of the array.

* :py:attr:`numpy.ndarray.shape`: the dimensions of the array (a tuple of integers indicating the size of the array in each dimension)

* :py:attr:`numpy.ndarray.dtype`: an object describing the type of the elements in the array.

.. doctest::
  :options: +NORMALIZE_WHITESPACE

  >>> print A.ndim
  2
  >>> print A.shape
  (2, 4)
  >>> print A.dtype # doctest: +SKIP
  int64

Accessing array elements
------------------------

The Python_ `operator[] <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_ 
allows to index the elements of an array. Please note that the indices start 
at 0.

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
----------

The type of the elements of an array can be specified at the creation time.

.. doctest::

  >>> C = numpy.array( [[1,2], [3,4]], dtype='float64' )
  >>> print C.dtype
  float64


If you would like to cast the elements of an array to another type you can
do this by using the NumPy_ function :py:attr:`numpy.ndarray.astype`.

.. doctest::

  >>> D = C.astype('uint8')
  >>> print D.dtype
  uint8

In addition, |project| provides the :py:func:`bob.core.array.convert` function 
which allows you to convert/rescale a NumPy_ :py:class:`numpy.ndarray` of a 
given type into another array, of possibly different type, with re-scaling.
Typically, this is useful if you want to convert a uint8 2D array (such as a
grayscale image) into a float64 2D array with a ``[0,1]`` range.

.. doctest::
  :options: +NORMALIZE_WHITESPACE

  >>> img = numpy.array([[0,1,2,3,4],[255,254,253,252,251]], dtype='uint8')
  >>> img_d = bob.core.array.convert(img, dtype='float64', dest_range=(0.,1.))
  >>> print img_d
  [[ 0. 0.00392157 0.00784314 0.01176471 0.01568627]
   [ 1. 0.99607843 0.99215686 0.98823529 0.98431373]]
  >>> print img_d.dtype
  float64


Array shape
-----------

NumPy_ provides several features to reshape or stack arrays, such as the
:py:attr:`numpy.ndarray.reshape`, :py:func:`numpy.hstack`, and
:py:func:`numpy.vstack` methods.

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
-----------------------

NumPy_ provides numerous mathematical operations. Most of them are performed
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


NumPy_ also provides reduction operations.

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

Linear algebra is also supported through the bridges to the optimized ATLAS_
LAPACK_ (and BLAS_) libraries which are mostly integrated in the `linalg` 
submodule of SciPy_. In the following, this is highlighted via two different
examples: matrix multiplication and matrix inversion.

.. doctest::
  :options: +NORMALIZE_WHITESPACE
   
  >>> A = numpy.array([[1,2],[3,4]]) # Creates a 2D array / matrix
  >>> B = numpy.array([[5,6],[7,8]]) # Creates a 2D array / matrix
  >>> C = numpy.dot(A,B) # Computes the matrix multiplication A*B
  >>> print C
  [[19 22]
   [43 50]]
  >>> import scipy.linalg
  >>> D = scipy.linalg.inv(C) # Computes the inverse of C
  >>> print D # doctest: +SKIP
  [[ 12.5   -5.5 ]
   [-10.75   4.75]]

Assignment, shallow and deep copy
---------------------------------

Different arrays might share the same data in memory. Let's first have a look
at the assignment operator "=".

.. doctest::
  :options: +NORMALIZE_WHITESPACE

  >>> a = numpy.array([1,2,3,4], dtype='uint8')
  >>> b = a # Asignment -> No copy at all
  >>> print b is a # a and b are two names for the same ndarray object
  True

The assignment operator only creates an **alias** to the same 
:py:class:`numpy.ndarray` object. In contrast, the 
:py:attr:`numpy.ndarray.view` method creates a new NumPy_ array object that
points to the same memory block. This is known as a **shallow copy**.

.. doctest::
  :options: +NORMALIZE_WHITESPACE

  >>> c = a.view()
  >>> print c is a  # a and c are two different ndarray objects
  False
  >>> c[2] = 7  # but they share the same data in memory
  >>> print a
  [1 2 7 4]

In a similar way, an :py:class:`numpy.ndarray` might be sliced, and in this 
case, the data are still shared between the two :py:class:`numpy.ndarray` 
instances.

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

If we would like to do a **deep copy**, we could use the 
:py:attr:`numpy.ndarray.copy` method.

.. doctest::
  :options: +NORMALIZE_WHITESPACE

  >>> e = a.copy() # e is a complete copy of a
  >>> e[0] = 7
  >>> print a
  [0 2 7 4]

For a more exhaustive introduction about NumPy_, please have a look at its 
`user guide`_. 

Digital signals as multi-dimensional arrays
===========================================

For |project|, we have decided to represent digital signals directly as 
NumPy_ arrays, rather than having dedicated classes for each type of 
signals. This implies that some convention has been defined.


Vectors and matrices
--------------------

A vector is represented as a 1D NumPy_ array, whereas a matrix is 
represented by a 2D array whose first dimension corresponds to the rows, and
second dimension to the columns.

.. doctest::
  :options: +NORMALIZE_WHITESPACE

  >>> A = numpy.array([[1, 2, 3], [4, 5, 6]], dtype='uint8') # A is a matrix 2x3
  >>> print A
  [[1 2 3]
   [4 5 6]]
  >>> b = numpy.array([1, 2, 3], dtype='uint8') # b is a vector of length 3
  >>> print b
  [1 2 3]

Images
------

**Grayscale** images are represented as 2D arrays, the first dimension being the
height (number of rows) and the second dimension being the width (number of 
columns).

For instance,

.. doctest::
  :options: +NORMALIZE_WHITESPACE

  >>> img = numpy.ndarray((480,640), dtype='uint8')

**img** which is a 2D array can be seen as a grayscale image of dimension
640 (width) by 480 (height). In addition, **img** can be seen as a matrix
with 480 rows and 640 columns. This is the reason why we have decided that for
images, the first dimension is the height and the second one the width, such
that it matches the matrix convention as well.

**Color** images are represented as 3D arrays, the first dimension being the 
number of color planes, the second dimension the height and the third the 
width. As an image is an array, this is the responsibility of the user to know
in which color space the content is stored. |project| provides functions to 
perform colorspace conversion (cf. the :doc:`TutorialsIP` tutorial).

Videos
------

A video can be seen as a sequence of images over time. By convention, the 
first dimension is for the frame indices (time index), whereas the remaining 
ones are related to the corresponding image frame.

Audio signal
------------

|project| does not yet support audio files (No wav or mp3 codec). However, 
there are several alternatives: 

* The signal in the audio file could be converted into HDF5. After this it can then
  be read and processed with |project|.

* A Python_ module could be used to load the audio file as a NumPy_ :py:class:`numpy.ndarray`, such as :py:mod:`scipy.io.wavfile`. Please look at :ref:`audiosignal` for an example of how to do it.

Once loaded, a stereo audio signal would be represented as a 2D array, the first 
dimension corresponding to the time index and the second one to the audio 
channel, values in the array would then correspond to wave magnitudes.

Interfacing with OpenCV_ and PIL_
=================================

As |project| relies on NumPy_ arrays, it is very easy to make use of other 
popular libraries such as OpenCV_ and PIL_.


OpenCV_
-------

.. note::

   The new `cv2` module of OpenCV_ 2.x is able to process NumPy_ arrays 
   directly, which makes the following conversions unnecessary.

To convert a NumPy_ array into an OpenCV_ cvMat, the 
:py:func:`cv.fromarray` method of OpenCV_ will do the job.

.. code-block:: python

  >>> import cv, numpy
  >>> a = numpy.ones((5, 10))
  >>> mat = cv.fromarray(a)

Similarly, to perform the inverse conversion from an OpenCV_ cvMat into a 
NumPy_ array, the :py:func:`numpy.asarray` method is suitable.

.. code-block:: python

  >>> mat = cv.CreateMat(3, 5, cv.CV_32FC1)
  >>> cv.Set(mat, 37)
  >>> a = numpy.asarray(mat)

Both NumPy_ array and OpenCV_ cvMat use similar datatypes (`uint8`, 
`uint32`, `float64`, etc.), and hence, it is interesting to notice that the 
datatype is preserved by the previous operations.

PIL_
----

PIL_ does not provide a generic multi-dimensional array structure. However, 
its Image structure can be seen as 2D or 3D arrays. To convert a 2D NumPy_ 
array of type `uint8` into a grayscale (integer) PIL_ image, the 
:py:func:`Image.fromarray` method of PIL_ will do the job.

.. code-block:: python

  >>> import numpy, Image
  >>> img = numpy.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]], 'uint8')
  >>> imgPIL = Image.fromarray(img)

To convert a grayscale PIL_ image into a 2D NumPy_ array of `uint8`, 
the :py:func:`numpy.asarray` method of NumPy_ is suitable.

.. code-block:: python

  >>> img2 = numpy.asarray(imgPIL)
  >>> numpy.array_equal(img, img2)
  True

In contrast to OpenCV_, please be aware that PIL_ does not support all the
datatypes that we have in |project|. Therefore, please restrict yourself to 
`uint8` (and `float32` for grayscale images) when you proceed with back and
forth conversions or take the time to check that your operations are really 
valid and expected.

Converting color images is more tricky as |project| uses plane color images
whereas PIL_ relies, by default, on interleaved color images. Therefore, there
is an additional conversion required.

.. code-block:: python

  >>> import numpy, Image
  >>> a = numpy.array([[[1,2,3],[4,5,6]],[[11,12,13],[14,15,16]],[[21,22,23],[24,25,26]]], 'uint8')
  >>> c = numpy.dstack((a[0,:],a[1,:],a[2,:])).reshape(a.shape[1],a.shape[2],a.shape[0]) # Convert to plane color to interleaved color
  >>> cPIL = Image.fromarray(c)

The reverse operation is similar, but again requires an extra conversion from
interleaved color image to plane color image.

.. code-block:: python

  >>> c_read = numpy.asarray(cPIL)
  >>> c_plane_read = numpy.vstack((c_read[:,:,0],c_read[:,:,1],c_read[:,:,2])).reshape(c_read.shape[2],c_read.shape[0],c_read.shape[1]) # Convert interleaved color to plane color
  >>> numpy.array_equal(a, c_plane_read)
  True

MATLAB_
-------

|project| currently does not provide a MATLAB_ mex interface. Nevertheless, it
is possible to load and save simple `.mat` files thanks to the MatIO_ 
library. However, complex data such as MATLAB_ structures are not supported.
Be aware that MATLAB_ also supports the HDF5_ file format. For more 
details, please have a look at :doc:`TutorialsIO`.

Random Number Generation
========================

We have developed a set of bridges to the `Boost Random Number Generation`_
facilities. This allows you to generate random numbers in a variety of ways.

.. code-block:: python

  >>> mt = bob.core.random.mt19937()
  >>> binom = bob.core.random.binomial_float64()
  >>> binom(mt)
  0 

.. note::

  `NumPy`_ also provides random sampling functionalities.

.. include:: links.rst

.. Place here your external references

.. _boost random number generation: http://www.boost.org/doc/libs/release/libs/random/index.html 
.. _user guide: http://docs.scipy.org/doc/numpy/user/
.. _pil: http://www.pythonware.com/products/pil/
.. _atlas: http://math-atlas.sourceforge.net/
.. _blas: http://www.netlib.org/blas/
.. _page: http://www.scipy.org/NumPy_for_Matlab_Users page
