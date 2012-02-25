.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Laurent El Shafey <laurent.el-shafey@idiap.ch>
.. Tue Apr 5 09:16:14 2011 +0200
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

==========================
 Multi-Dimensional Arrays
==========================

The basic data structure of |project| are multi-dimensional arrays. They
are indeed suitable representations for various types of signals: A color
image can be represented by a 3D array, MFCCs extracted on a speech sequence
can be represented by a 2D array, etc.

At the C++ level, the Blitz_ library is used to handle arrays. Although we 
initially binds Blitz_ Arrays into Python, we quickly realized that it might 
be more clever to use existing NumPy_ ndarrays from Python, as they can
directly be processed by numerous existing Python libraries such as NumPy_ 
and SciPy_. 

This means that |project| multi-dimensional arrays are represented in Python 
by NumPy_ ndarrays. This also implies that there are internal conversion 
routines to convert NumPy_ ndarrays from/to Blitz_. As they are done 
implicitly, the user has no need to care about this aspect and should just 
use NumPy_ ndarrays everywhere.

For an introduction and tutorials about NumPy_ ndarrays, just 
visit the Numpy_ website.

Supported Types
---------------

Because Blitz_ arrays are template types, we must fix the set of supported
combinations of element type and number of dimensions that are supported by
|project| and subsequently at the python bridge. Indeed, Blitz_ arrays a
flexible containers that support a different range of base element types and 
up to 11 dimensions. At |project|, we limit the support of provided code to
manipulate arrays *up to 4 dimensions* and using the following element types.
These element types and ranks should cover all of the functionality needed by
machine learning algorithms and signal processing utilities.

+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| Common name  |   C++                       | `NumPy`_     | Meaning                                                      |
+==============+=============================+==============+==============================================================+
| bool         | bool                        | bool         | a boolean                                                    |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| int8         | int8_t                      | int8         | a signed integer, with 8 bits                                |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| int16        | int16_t                     | int16        | a signed integer, with 16 bits                               |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| int32        | int32_t                     | int32        | a signed integer, with 32 bits                               |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| int64        | int64_t                     | int64        | a signed integer, with 64 bits                               |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| uint8        | uint8_t                     | uint8        | a unsigned integer, with 8 bits                              |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| uint16       | uint16_t                    | uint16       | a unsigned integer, with 16 bits                             |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| uint32       | uint32_t                    | uint32       | a unsigned integer, with 32 bits                             |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| uint64       | uint64_t                    | uint64       | a unsigned integer, with 64 bits                             | 
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| float32      | float                       | float32      | a real number with single precision, with 32 bits            |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| float64      | double                      | float64      | a real number with double precision, with 64 bits            |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| complex64    | std::complex<float>         | complex64    | a complex number with each part using each a 32-bit float    |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| complex128   | std::complex<double>        | complex128   | a complex number with each part using each a 64-bit float    |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+

.. dropped support:
  +--------------+-----------------------------+--------------+--------------------------------------------------------------+
  | ~~float128~~ | ~~long double~~             | ~~float128~~ | ~~a real number with quadruple precision, with 128 bits~~    |
  +--------------+-----------------------------+--------------+--------------------------------------------------------------+
  | -complex256- | -std::complex<long double>- | -complex256- | -a complex number with each part using each a 128-bit float- |
  +--------------+-----------------------------+--------------+--------------------------------------------------------------+

.. note::

  We have decided not to support ``long double`` and
  ``std::complex<long double>`` because of variations between 32 and 64-bit
  architectures in the representation of ``long double``'s. A simple 
  experiment with gdb can show you the problem. Under a 32-bit Linux machine,
  the sizes of the types listed above are:

  .. code-block:: none
  
    Size of built-in types: 
    bool: 1
    uint8_t: 1
    uint16_t: 2
    uint32_t: 4
    uint64_t: 8
    int8_t: 1
    int16_t: 2
    int32_t: 4
    int64_t: 8
    float: 4
    double: 8
    long double: 12
    std::complex<float>: 8
    std::complex<double>: 16
    **std::complex<long double>: 24**

    size_t: 4
    unsigned int: 4
    int: 4

  Whereas in a 64-bit machine, this is what you would see:

  .. code-block:: none
  
    Size of built-in types: 
    bool: 1
    uint8_t: 1
    uint16_t: 2
    uint32_t: 4
    uint64_t: 8
    int8_t: 1
    int16_t: 2
    int32_t: 4
    int64_t: 8
    float: 4
    double: 8
    long double: 12
    std::complex<float>: 8
    std::complex<double>: 16
    **std::complex<long double>: 32**

    size_t: 8
    unsigned int: 4
    int: 4

  This made it hard to write code that can I/O data properly. Moreover, long
  doubles are not widely popular, making this choice an easy one.

Convert function
-----------------

The convert() function allows to convert/rescale an array of a given type 
into another array of a possibly different type with re-scaling. Typically,
this is useful if we want to convert a uint8 2D array (e.g. a grayscale image)
into a float64 2D array with a [0,1] range.

.. code-block:: python

  >>> import numpy
  >>> img = numpy.array([[0,1,2,3,4],[255,254,253,252,251]], dtype='uint8')
  >>> bob.core.array.convert(img, dtype='float64', destRange=(0.,1.))
  >>> img_d = bob.core.array.convert(img, dtype='float64', destRange=(0.,1.))
  >>> print img_d
      [[ 0.          0.00392157  0.00784314  0.01176471  0.01568627]
      [ 1.          0.99607843  0.99215686  0.98823529  0.98431373]]
  >>> print img_d.dtype
      float64



.. Place here your references
.. _blitz: http://www.oonumerics.org/blitz
.. _numpy: http://numpy.scipy.org
.. _scipy: http://www.scipy.org
.. _here: http://www.idiap.ch/software/bob/wiki/bobDatabaseBindata#Binaryfileformatheader
