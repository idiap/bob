.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed Jun 22 17:50:08 2011 +0200
.. 
.. Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

Input and output in |project| is focused around three primitives:

:py:class:`bob.io.Array`

  The Array represents a :cpp:type:`blitz::Array` abstraction. It can hold
  either a real :cpp:type:`blitz::Array` representation or be just a
  pointer to a datafile where the actual array can be loaded from.

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

Data Description Language
-------------------------

It is often useful to talk about data as represented in a file. HDF5_ uses a
neat descriptive language for this purpose, called Data Description Language or
DDL_ for short. If necessary, we will use this descriptive format to talk about
our examples and how bob will encode and decode the data.

HDF5 Standard Utilities
-----------------------

It is often useful to list, dump or check differences in files. For this
purpose, recommend extensive use of simple command line utilities from the
HDF5_ project:

h5dump
  Dumps the whole contents of the file using the DDL

h5ls
  Lists the contents of the file using DDL, do not show the data

h5diff
  Finds differences in HDF5_ files.

Saving Data
-----------

Let's take a look on how to record simple scalar data such as integers or
floats.

.. code-block:: python

  import bob
  an_integer = 5
  a_float = 3.1416
  f = bob.io.HDF5File('example1.hdf5')
  f.set('my_integer', an_integer)
  f.set('my_float', a_float)
  del f

This example shows how you how to create a |project|
:py:class:`bob.io.HDF5File` and set two values on it. If you use the HDF5_
command line utility ``h5dump`` on the file ``example1.hdf`` you will verify
the file now contains:

.. code-block:: none

  HDF5 "example1.hdf5" {
  GROUP "/" {
     DATASET "my_float" {
        DATATYPE  H5T_IEEE_F64LE
        DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
        DATA {
        (0): 3.1416
        }
     }
     DATASET "my_integer" {
        DATATYPE  H5T_STD_I32LE
        DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
        DATA {
        (0): 5
        }
     }
  }
  }


.. note::

  All our types and methods are defined in the ``bob`` package. We omit this
  on the following examples.

  We delete the ``HDF5File`` object in the end of the above script, to make
  sure the file will be safely closed and all buffers flushed. You don't need
  to do that explicetly in your script. As soon as ``HDF5File`` objects go out
  of scope, proper flushing and closing will take place. We will, therefore,
  omit this on the following examples.

.. note::

  In |project|, when you open an HDF5File, you can choose one of the following
  flags:

  'r'
    Open the file only to read. Write operations will fail
  'w'
    Open the file to read and write with appending (this is the default!)
  't'
    Open the file to read and write, but truncate it
  'x'
    Read/write/append with exclusive access

The dump shows that there are two datasets inside a group named ``/`` the file.
HDF5 groups are like filesystem directories. They create namespaces for the
data. In the root group (or directory), we find our two variables, named as we
set them to be.  The variable names are the complete path to the location where
they live. We could write a new variable in the same file, but in a different
directory like this:

.. code-block:: python

  f = bob.io.HDF5File('example1.hdf5', 'w')
  f.set('/test/my_float', 6.28, dtype='float32')

Line 1 shows we open the file again for reading and writing, but without
truncating it. This will allow us to access the file contents. Next, we write a
new variable inside the ``/test`` subdirectory. As you can verify, **for simple
scalars**, we can also force the storage type. Where normally one would have a
64-bit real value, we impose that this variable is saved as a 32-bit real
value. You can verify the dump correctness with ``h5dump``:

.. code-block:: none

  GROUP "/" {
  ...
   GROUP "test" {
      DATASET "my_float" {
         DATATYPE  H5T_IEEE_F32LE
         DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
         DATA {
         (0): 6.28
         }
      }
   }
  }

Notice the subdirectory ``test`` has been created and inside it a floating
point number has been stored. Such a float point number has a 32-bit precision
as it was defined.

.. note::

  If you need to place lots of variables in a subfolder, it may be better to
  setup the prefix folder before starting the writing operations on the
  ``HDF5File`` object. You can do this using the method ``HDF5File.cd``.
  Look-up its help for more information and usage instructions.

Writing arrays is a little simpler as ``blitz::Array<>``'s encode all the type
information we need to write and read them correctly. Here is an example:

.. code-block:: python

  a = bob.core.array.int8_2(range(4), (2,2))
  f.set('my_array', a)

And the resulting ``h5dump`` would be:

.. code-block:: none

  ...
   DATASET "my_array" {
      DATATYPE  H5T_STD_I8LE
      DATASPACE  SIMPLE { ( 2, 2 ) / ( 2, 2 ) }
      DATA {
      (0,0): 0, 1,
      (1,0): 2, 3
      }
   }
  ...

You don't need to limit yourself to single variables, you can also save lists
of scalars and arrays using ``append`` instead of ``set``. Try it!

Reading Data
------------

Reading up data you just wrote is as easy. For this task you should use
:py:meth:`bob.io.HDF5File.read`. The read method will read all the
contents of the variable pointed by the given path. This is the normal way to
read a variable you have written with :py:meth:`bob.io.HDF5File.set()``. If
you decided to create a list of scalar or arrays, the way to read that up would
be using :py:meth:`bob.io.HDF5File.lread()` instead. Here is an example:

.. code-block:: python

  >>> f = bob.io.HDF5File('example1.hdf5', 'r') #read only
  >>> f.read('my_integer') #reads integer
  5
  >>> f.read('my_float') # reads float
  3.141599999999999
  >>> print f.read('my_array') # reads the array
  [[0 1]
   [2 3]]

Now let's look at an example where we have used
:py:meth:`bob.io.HDF5File.append()` instead of
:py:meth:`bob.io.HDF5File.set()` to write data to a file. That is normally
the case when you write lists of variables to a dataset.

.. code-block:: python

  >>> f = bob.io.HDF5File('example2.hdf5')
  >>> f.append('arrayset', bob.core.array.float64_1(range(10),(10,)))
  >>> f.append('arrayset', 2*bob.core.array.float64_1(range(10),(10,)))
  >>> f.append('arrayset', 3*bob.core.array.float64_1(range(10),(10,)))
  >>> print f.lread('arrayset', 0)
  [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
  >>> print f.lread('arrayset', 2)
  [  0.   3.   6.   9.  12.  15.  18.  21.  24.  27.]

This is how a ``h5dump`` of the file looks like:

.. code-block:: none

  HDF5 "example2.hdf5" {
  GROUP "/" {
     DATASET "arrayset" {
        DATATYPE  H5T_IEEE_F64LE
        DATASPACE  SIMPLE { ( 3, 10 ) / ( H5S_UNLIMITED, 10 ) }
        DATA {
        (0,0): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        (1,0): 0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
        (2,0): 0, 3, 6, 9, 12, 15, 18, 21, 24, 27
        }
     }
  }
  }
  
Notice that the expansion limits for the first dimension have been correctly
set by |project| so you can insert an *unlimited* number of 1D float vectors.
Of course, you can also read the whole contents of the arrayset in a single
shot:

.. code-block:: python

  >>> print f.read('arrayset')
  [[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.]
   [  0.   2.   4.   6.   8.  10.  12.  14.  16.  18.]
   [  0.   3.   6.   9.  12.  15.  18.  21.  24.  27.]]
  
As you can see, the only difference between :py:meth:`bob.io.HDF5File.read()`
and :py:meth:`bob.io.HDF5File.lread()` is on how |project| considers the
available data (as a single array with N dimensions or set of arrays with N-1
dimensions). In the first example, you would have also been able to read the
variable `my_array` as an arrayset using :py:meth:`bob.io.HDF5File.lread()`
instead of :py:meth:`bob.io.HDF5File.read()`. In this case, each position
readout would return a 1D uint8 array instead of a 2D array.

Array interfaces
----------------

What we have shown so far is the generic API to read and write data using HDF5.
You will use it when you want to import or export data from |project| into
other software frameworks, debug your data or just implement your own classes
that can serialize and de-serialize from HDF5 file containers. In |project|,
most of the time you will be working with :py:class:`bob.io.Array`\s and
:py:class:`bob.io.Arrayset`\s and it is even simpler to load and save those
from/to files. 

To create an :py:class:`bob.io.Array` from a file, just do the following:

.. code-block:: python

  >>> a = bob.io.Array('array.hdf5')
  >>> a.filename
  'array.hdf5'
  >>> a.loaded
  False

Arrays are containers for :cpp:class:`blitz::Array`\s **or** just pointers
to a file.  When you instantiate an :py:class:`bob.io.Array` it does **not**
load the file contents into memory. It waits until you emit another explicit
instruction to do so. We do this with the :py:meth:`bob.io.Array.get()`
method:

.. code-block:: python

  >>> bzarray = a.get()
  blitz::Array<float64,2> (3, 10) (0x1044a0488)
  >>> bzarray[0,0]
  >>> bzarray[0,0] = -1
  >>> print bzarray
  [[ -1.   1.   2.   3.   4.   5.   6.   7.   8.   9.]
   [  0.   2.   4.   6.   8.  10.  12.  14.  16.  18.]
   [  0.   3.   6.   9.  12.  15.  18.  21.  24.  27.]]

Every time you say :py:meth:`bob.io.Array.get()`, the file contents will be
read from the file and into a new array. Try again:

.. code-block:: python

  >>> a.loaded
  False
  >>> bzarray = a.get()
  >>> bzarray
  blitz::Array<float64,2> (3, 10) (0x1459a99b0)
  >>> print bzarray[0,0]
  0.0

You can force permanently loading the contents of the file in memory an avoid
the I/O costs every time you read issue a :py:meth:`bob.io.Array.get()`:

.. code-block:: python

  >>> a.load() #move contents to memory
  >>> a.loaded
  True
  >>> a.filename
  ''
  >>> bzarray = a.get()
  >>> bzarray[0,0] = -1
  >>> bzarray_reference = a.get()
  >>> print bzarray_reference[0,0]
  -1.0

Notice that, once the array is loaded in memory, a reference to the same array
is shared every time you call :py:meth:`bob.io.Array.get()`.

Saving the :py:class:`bob.io.Array` is as easy, just call the
:py:meth:`bob.io.Array.save()` method:

.. code-block:: python

  >>> a.save('copy.hdf5')

Blitz Array Shortcuts
=====================

To just load a :cpp:class:`blitz::Array` in memory, we have written a
short cut that lives at :py:func:`bob.core.array.load` and saves you from
going through the :py:class:`bob.io.Array` API:

.. code-block:: python

  >>> t = bob.core.array.load('example2.hdf5')
  >>> t
  blitz::Array<float64,2> (3, 10) (0x11023b410)
  >>> print t
  [[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.]
   [  0.   2.   4.   6.   8.  10.  12.  14.  16.  18.]
   [  0.   3.   6.   9.  12.  15.  18.  21.  24.  27.]]

You can also directly save :cpp:class:`blitz::Array`\s without going
through the :py:class:`bob.io.Array` container:

.. code-block:: python

  >>> t.save('copy.hdf5')

.. note::

  Under the hood, we still use the :py:class:`bob.io.Array` API to execute
  the read and write operations. This avoids code duplication and hooks data
  loading and saving to the powerful |project| transcoding framework that is
  explained next.

Array transcoding
=================

|project| would not be this useful if we could only read and write HDF5 files.
For this reason, we provide support to load and save data from many different
file types including Matlab ``.mat`` files, various image file types and video
data. File types and specific serialization and de-serialization is switched
automatically using filename extensions. These are the extensions and formats
we currently support:

* Images: return RGB (uint8_t or uint16_t) 3-D arrays, or Grayscale (uint8_t or
  uint16_t) 2-D arrays as indicated [``bob.image``]. Please notice that for
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
  video file. The following extensions are supported:

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

  >>> bzarray.save('data.mat')

You can choose an unconforming extension, but then make sure to choose the
right codec as defined before:

.. code-block:: python

  >>> bzarray.save('data.myweird.extension', 'matlab.array.binary')
  >>> #data is saved in Matlab format

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
  >>> len(s)
  0
  >>> s.append(bob.core.array.array(range(5), 'float32'))
  >>> len(s)
  1
  >>> s.elementType
  libpybob_io.ElementType.float32

This can quickly become inefficient if you have many arrays to pull in.
Instead, you can extend the arrayset with a list of arrays pretty much like you
do with a normal python list:

.. code-block:: python

  >>> t = bob.core.array.array(range(5), 'float32')
  >>> s.extend([t,t,t,t,t,t,t])
  >>> len(s)
  8

Optionally, you can also extend the Arrayset with a
:cpp:class:`blitz::Array` object with N dimensions and tell it to iterate
through dimension D and add objects with N-1 dimensions to it:

.. code-block:: python

  >>> t = bob.core.array.float32_2(range(50), (10,5))
  >>> s.extend(t, 0)
  >>> len(s)
  18

The above code will quickly all all rows from ``t`` to ``s``. It is equivalent
to the python code:

.. code-block:: python

  >>> for k in range(t.extent(0)): 
  ...   s.append(t[k,:])
  >>> len(s)
  28

The only difference being it is executed in pure C++ and is therefore, much
faster than individually appending each sub-array.

Transcoding (binary) files
--------------------------

Transcoding is the operation of converting files saved in one (binary) format
to another. You can transcode from/to any of the types described above, as long
as the underlying blitz::Array remains compatible with the chosen format. For
example, you can save a JPEG image as a |project| (``.hdf5``) file. You cannot
save a complex array inside a |project| (``.hdf5``) file into a bob3
(``.bindata``) simply because it only accepts single or double precision float
numbers.

|project| provides scripts that implements the above with a few bells and
whistles:

.. code-block:: sh

  $ array_transcode.py from-file to-file
  # or
  $ arrayset_transcode.py from-file to-file

If you execute these scripts without any parameters, an usage instruction and a
**list of built-in codecs** will be displayed.

Reference
---------

.. automodule:: bob.io
  :members:

.. Place your references here

.. _hdf5: http://www.hdfgroup.org/HDF5
.. _ddl: http://www.hdfgroup.org/HDF5/doc/ddl.html
