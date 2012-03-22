.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed Jun 22 17:50:08 2011 +0200
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

.. Index file for the Python bob::io bindings

===================
 Data Input/Output
===================

.. todo::

  This section requires a major revision: move stuff that matters to the
  relevant tutorial.

Input and output in |project| is focused around three primitives:

:py:class:`bob.io.Array`

  The `bob.io.Array` is an I/O capable version of :py:class:`numpy.ndarray`
  objects.

:py:class:`bob.io.Arrayset`

  The Arrayset represents a collection of :py:class:`bob.io.Array`'s.

:py:class:`bob.io.HDF5File`

  The HDF5File is our preferred container for all sorts of array and scalar
  data to be produced or consumed from |project|. HDF5_ is a flexible
  open-source scientific data storage format. |project| uses this format as the
  primary input and output format. HDF5_ packaged distributions are available
  for all major operation systems. Most commercial and open-source scientific
  platforms also support HDF5_ making it dead simple to import and export data
  from and to |project|.

  .. note::

    HDF5_ files are normally sufixed with either a ``.hdf5`` or `.h5`
    extensions.

Array transcoding
=================

|project| would not be this useful if we could only read and write HDF5 files.
For this reason, we provide support to load and save data from many different
file types including Matlab ``.mat`` files, various image file types and video
data. File types and specific serialization and de-serialization is switched
automatically using filename extensions. These are the extensions and formats
we currently support:

* Images: return RGB (uint8 or uint16) 3D arrays, or Grayscale (uint8 or
  uint16) 2-D arrays as indicated [``bob.image``]. Please notice that for
  this Array codec, file extensions DO matter even if a codecname is specified,
  as they are also used by ImageMagick to select the image loader/writer. The
  following extensions are supported:

  * bmp: RGB, bitmap format
  * gif: RGB, GIF
  * jpeg: RGB, Joint-Photograph Experts Group
  * pbm: Grayscale, Portable binary map (images)
  * pgm: Grayscale, Portable grayscale map
  * png: RGB, Portable network graphics, indexed
  * ppm: RGB, Portable pixel map
  * tiff: RGB
  * xcf: RGB, Gimp native format (**loading only**)

* Videos: returns a sequence of frames (loaded in memory) for all data within a
  video file. Returns 3D uint8 arrays. The following extensions are supported:

  * avi
  * dv
  * filmstrip
  * flv
  * h261
  * h263
  * h264
  * mov
  * image2
  * image2pipe
  * m4v
  * mjpeg
  * mpeg
  * mpegts
  * ogg
  * rawvideo
  * rm
  * rtsp
  * yuv4mpegpipe

* Other binary formats: 
  
  * Matlab (``.mat``), Matlab arrays, supports all integer, float and complex varieties [``matlab.array.binary``];
  * bob3 (``.bindata``), supports single or double precision float numbers, only 1-D [``bob3.array.binary``];
  * bob beta (``.bin``), supports all element types in |project| and any dimensionality [``bob.array.binary``] (*deprecated*);
  * bob alpha (``.tensor``) [``tensor.array.binary``] (*deprecated*);
  * **HDF5** (``.hdf5`` or ``.h5``) [``hdf5.array.binary``], is the **prefered
    format for enconding |project| data** as discussed before.

Saving a :py:class:`bob.io.Array` in a different format is just a matter of
choosing the right extension:

.. code-block:: python

  >>> bob.io.save(array, 'data.mat')

The extension chosen defines the recording format. For an overview of all
extensions and codecs supported with your version of |project|, you can execute
the command-line utitlity `info_table.py`:

.. code-block:: sh

  $ <bob-root>/bin/shell.py -- info_table.py

Arrayset interfaces
-------------------

Arraysets are lists of arrays with the same shape and element type. You can
load and save :py:class:`bob.io.Arrayset`\s pretty much like you do for a
plain :py:class:`bob.io.Array`\s.  All constraints and details for data
loading hold for this class as well such as the deferred loading property we
explained earlier.  A list of arrays can only be serialized and de-serialized
from specific formats:

  * Matlab (``.mat``), supporting all kinds of scalars and array types, with
    the codec ``matlab.arrayset.binary``;
  * bob3 (``.bindata``), supporting single or double precision float numbers,
    with the codec ``bob3.arrayset.binary``;
  * bob beta (``.bin``), supports all element types in |project| and any
    dimensionality, with the codec ``bob.arrayset.binary`` (*deprecated*);
  * bob alpha (``.tensor``), with the codec ``tensor.arrayset.binary``
    (*deprecated*);
  * **HDF5** (``.hdf5`` or ``.h5``), with the codec ``hdf5.arrayset.binary``,
    is the **prefered format for enconding |project| data** as discussed
    before.

Load and save operations
========================

Loading an arrayset is pretty much like loading an array, but instead,
internally, the code will use something like
:py:meth:`bob.io.HDF5File.lread()` automatically for you:

.. code-block:: python

  >>> s = bob.io.Arrayset('example2.hdf5')
  >>> print s
  <Arrayset[3] float64@(10,)>
  >>> len(s)
  3
  >>> s.filename
  '.../example2.hdf5'
  >>> s.loaded
  False
  >>> s.load()
  >>> s.loaded
  True
  >>> s.filename
  ''
  >>> s.save('transcoded.mat')

Building functionality
======================

The loading and saving functionality are a bit of old news... Let's look into
more interesting operations one can do with an :py:class:`bob.io.Arrayset`,
specifically to build one from ground up.

To build an Arrayset, you may call :py:meth:`bob.io.Arrayset.append()` as
many times you need:

.. code-block:: python

  >>> s = bob.io.Arrayset()
  >>> s.elementType
  libpybob_io.ElementType.unknown
  >>> print s.dtype #this is the numpy equivalent
  None
  >>> len(s)
  0
  >>> s.append(numpy.array(range(5), 'float32'))
  >>> len(s)
  1
  >>> s.elementType
  libpybob_io.ElementType.float32
  >>> s.dtype
  dtype('float32')

This can quickly become inefficient if you have many arrays to pull in.
Instead, you can extend the arrayset with a list of arrays pretty much like you
do with a normal python list:

.. code-block:: python

  >>> t = numpy.array(range(5), 'float32')
  >>> s.extend([t,t,t,t,t,t,t])
  >>> len(s)
  8

Optionally, you can also extend the Arrayset with a
:py:class:`numpy.ndarray` object with N dimensions and tell it to iterate
through dimension D and add objects with N-1 dimensions to it:

.. code-block:: python

  >>> t = numpy.array(range(50), 'float32').reshape(10,5)
  >>> s.extend(t, 0)
  >>> len(s)
  18

The above code will quickly append all rows from ``t`` to ``s``. It is
equivalent to the python code:

.. code-block:: python

  >>> for k in range(t.shape[0]): 
  ...   s.append(t[k,:])
  >>> len(s)
  28

The only difference being it is executed in pure C++ and is therefore much
faster than individually appending each sub-array.

Reference
---------

.. automodule:: bob.io

.. Place your references here

.. _hdf5: http://www.hdfgroup.org/HDF5
.. _ddl: http://www.hdfgroup.org/HDF5/doc/ddl.html
