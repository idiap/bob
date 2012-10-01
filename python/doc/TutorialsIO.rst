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

**************
 Input/Output
**************

This section gives an overview of the operations for storing and retrieving the basic data 
structures in |project|, such as `NumPy`_ arrays. |project| uses `HDF5`_  format for storing 
binary coded data. Using the |project| support for `HDF5`_, it is very simple to import and 
export data.

`HDF5`_  uses a neat descriptive language for representing the data in the HDF5 files, called 
Data Description Language (`DDL`_).

To perform the functionalities given in this section, you should 
have `NumPy`_ and |project| loaded into the `Python`_ environment.

.. testsetup:: *

   import numpy
   import bob
   import tempfile
   import os

   current_directory = os.path.realpath(os.curdir)
   temp_dir = tempfile.mkdtemp(prefix='bob_doctest_')
   os.chdir(temp_dir)
  
HDF5 standard utilities
=======================
Before explaining the basics of reading and writing to `HDF5`_ files, it is important to list 
some `HDF5`_ standard utilities for checking the content of an `HDF5`_ file. These are 
supplied by the `HDF5`_ project.

``h5dump``
  Dumps the content of the file using the DDL.

``h5ls``
  Lists the content of the file using DDL, but does not show the data.

``h5diff``
  Finds the differences between HDF5 files.


I/O operations using the class `bob.io.HDF5File`
================================================

Writing operations
------------------

Let's take a look at how to write simple scalar data such as integers or floats.

.. doctest::

   >>> an_integer = 5
   >>> a_float = 3.1416
   >>> f = bob.io.HDF5File('testfile1.hdf5', 'w')
   >>> f.set('my_integer', an_integer)
   >>> f.set('my_float', a_float)
   >>> del f

If after this you use the **h5dump** utility on the file ``testfile1.hdf5``, you will verify that the file now contains:

.. code-block:: none

  HDF5 "testfile1.hdf5" {
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

   In |project|, when you open a HDF5 file, you can choose one of the following options:
  
   **'r'** Open the file in reading mode; writing operations will fail (this is the default).

   **'a'** Open the file in reading and writing mode with appending.
 
   **'w'** Open the file in reading and writing mode, but truncate it.

   **'x'** Read/write/append with exclusive access.

The dump shows that there are two datasets inside a group named ``/`` in the file.
HDF5 groups are like file system directories. They create namespaces for the
data. In the root group (or directory), you will find the two variables, named as 
you set them to be.  The variable names are the complete path to the location where
they live. You could write a new variable in the same file but in a different
directory like this:

.. doctest::

  >>> f = bob.io.HDF5File('testfile1.hdf5', 'a')
  >>> f.create_group('/test')
  >>> f.set('/test/my_float', numpy.float32(6.28))
  >>> del f


Line 1 opens the file for reading and writing, but without truncating it. This will 
allow you to access the file contents. Next, the directory ``/test`` is created 
and a new variable is written inside the subdirectory. As you can verify, **for simple
scalars**, you can also force the storage type. Where normally one would have a
64-bit real value, you can impose that this variable is saved as a 32-bit real
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
  :py:class:`bob.io.HDF5File` object. You can do this using the method :py:meth:`HDF5File.cd()`.
  Look up its help for more information and usage instructions.

Writing arrays is a little simpler as the :py:class:`numpy.ndarray` objects encode 
all the type information we need to write and read them correctly. Here is an example:

.. doctest::

  >>> A = numpy.array(range(4), 'int8').reshape(2,2)
  >>> f = bob.io.HDF5File('testfile1.hdf5', 'a')
  >>> f.set('my_array', A)
  >>> del f

The result of running ``h5dump`` on the file ``testfile3.hdf5`` should be:

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
of scalars and arrays using the function :py:meth:`bob.io.HDF5.append()` instead 
of :py:meth:`bob.io.HDF5.set()`.

Reading operations
------------------

Reading data from a file that you just wrote to is just as easy. For this task you should use
:py:meth:`bob.io.HDF5File.read`. The read method will read all the
contents of the variable pointed to by the given path. This is the normal way to
read a variable you have written with :py:meth:`bob.io.HDF5File.set()`. If
you decided to create a list of scalar or arrays, the way to read that up would
be using :py:meth:`bob.io.HDF5File.lread()` instead. Here is an example:

.. doctest::

  >>> f = bob.io.HDF5File('testfile1.hdf5') #read only
  >>> f.read('my_integer') #reads integer
  5
  >>> print f.read('my_array') # reads the array
  [[0 1]
   [2 3]]
  >>> del f

Now let's look at an example where we have used
:py:meth:`bob.io.HDF5File.append()` instead of
:py:meth:`bob.io.HDF5File.set()` to write data to a file. That is normally
the case when you write lists of variables to a dataset.

.. doctest::

  >>> f = bob.io.HDF5File('testfile2.hdf5', 'w')
  >>> f.append('arrayset', numpy.array(range(10), 'float64'))
  >>> f.append('arrayset', 2*numpy.array(range(10), 'float64'))
  >>> f.append('arrayset', 3*numpy.array(range(10), 'float64'))
  >>> print f.lread('arrayset', 0)
  [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
  >>> print f.lread('arrayset', 2)
  [  0.   3.   6.   9.  12.  15.  18.  21.  24.  27.]
  >>> del f

This is what the ``h5dump`` of the file would look like:

.. code-block:: none

  HDF5 "testfile4.hdf5" {
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

.. doctest::
  
  >>> f = bob.io.HDF5File('testfile2.hdf5')
  >>> print f.read('arrayset')
  [[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.]
   [  0.   2.   4.   6.   8.  10.  12.  14.  16.  18.]
   [  0.   3.   6.   9.  12.  15.  18.  21.  24.  27.]]
  
As you can see, the only difference between :py:meth:`bob.io.HDF5File.read()`
and :py:meth:`bob.io.HDF5File.lread()` is on how |project| considers the
available data (as a single array with N dimensions or as a set of arrays with N-1
dimensions). In the first example, you would have also been able to read the
variable `my_array` as an arrayset using :py:meth:`bob.io.HDF5File.lread()`
instead of :py:meth:`bob.io.HDF5File.read()`. In this case, each position
readout would return a 1D uint8 array instead of a 2D array.
 
Array interfaces
================

What we have shown so far is the generic API to read and write data using HDF5.
You will use it when you want to import or export data from |project| into
other software frameworks, debug your data or just implement your own classes
that can serialize and de-serialize from HDF5 file containers. In |project|,
most of the time you will be working with :py:class:`numpy.ndarrays`\s and, in
rarer moments, with :py:class:`bob.io.Array`\s. It is easy to load and save
both types from files.

To create an :py:class:`bob.io.Array` from a file, just do the following:

.. doctest::

  >>> a = bob.io.Array('testfile2.hdf5')
  >>> a.filename
  'testfile2.hdf5'

Arrays are containers for :py:class:`numpy.ndarray`\s **or** just pointers
to a file.  When you instantiate an :py:class:`bob.io.Array` it does **not**
load the file contents into memory. It waits until you emit another explicit
instruction to do so. We do this with the :py:meth:`bob.io.Array.get()`
method:

.. doctest::

  >>> array = a.get()
  >>> array
  array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
         [  0.,   2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.],
         [  0.,   3.,   6.,   9.,  12.,  15.,  18.,  21.,  24.,  27.]])

Every time you say :py:meth:`bob.io.Array.get()`, the file contents will be
read from the file and into a new array. Try again:

.. doctest::

  >>> array = a.get()
  >>> array
  array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
         [  0.,   2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.],
         [  0.,   3.,   6.,   9.,  12.,  15.,  18.,  21.,  24.,  27.]])


Calling :py:meth:`bob.io.Array.get()` will, by default, load the contents of the
file into memory. Doing this means that subsequent calls to :py:meth:`bob.io.Array.get()`
avoid having to read the file again and so do not incur an additional I/O cost.

.. doctest::

  >>> a.load() #move contents to memory
  >>> a.filename
  ''
  >>> array = a.get() # if you do 'get()' again, you will get a reference to same object!
  >>> array_reference = a.get()
  >>> print array_reference[0,0]
  0.0

Notice that, once the array is loaded in memory, a reference to the same array
is shared every time you call :py:meth:`bob.io.Array.get()`.

Saving the :py:class:`bob.io.Array` is as easy, just call the
:py:meth:`bob.io.Array.save()` method:

.. doctest::

  >>> a.save('copy1.hdf5')

Numpy ndarray shortcuts
=======================

To just load an :py:class:`numpy.ndarray` in memory, you can use a short cut that
lives at :py:func:`bob.io.load`. This short cut means that you don't have to go 
through the :py:class:`bob.io.Array` container:

.. doctest::

  >>> t = bob.io.load('testfile2.hdf5')
  >>> t
  array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
         [  0.,   2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.],
         [  0.,   3.,   6.,   9.,  12.,  15.,  18.,  21.,  24.,  27.]])

You can also directly save :py:class:`numpy.ndarray`\s without going
through the :py:class:`bob.io.Array` container:

.. doctest::

  >>> bob.io.save(t, 'copy2.hdf5')

.. note::

  Under the hood, we still use the :py:class:`bob.io.Array` API to execute
  the read and write operations. This avoids code duplication and hooks data
  loading and saving to the powerful |project| transcoding framework that is
  explained next. 

Reading and writing image and video data
========================================

|project| provides support to load and save data from many different
file types including Matlab ``.mat`` files, various image file types and video
data. File types and specific serialization and de-serialization is switched
automatically using filename extensions. Knowing this, saving an array in a different format is just a matter of
choosing the right extension. This is illustrated in the following example, where an image generated randomly using the 
method `NumPy` :py:meth:`numpy.random.random_integers()`, is saved in JPEG format. The image must be of type uint8 or uint16.

.. doctest::

  >>> my_image = numpy.random.random_integers(0,255,(3,256,256))
  >>> bob.io.save(my_image.astype('uint8'), 'testimage.jpg') # saving the image in jpeg format
  >>> my_copy_image = bob.io.load('testimage.jpg')

As for reading the video files, although it is possible to read a video using the :py:meth:`bob.io.load()`, you should use the 
methods of the class :py:class:`bob.io.VideoReader` to read frame by frame and avoid overloading your machine's memory. In the following 
example you can see how to create a video and save it using the class :py:class:`bob.io.VideoWriter` and load it again using the 
class :py:class:`bob.io.VideoReader`. The created video will have 30 frames generated randomly. Due 
to *FFMPEG* constrains, the width and height of the video need to be multiples of two.

.. doctest::

  >>> width = 50; height = 50;
  >>> framerate = 24
  >>> outv = bob.io.VideoWriter('testvideo.avi', height, width, framerate) # output video
  >>> for i in range(0, 30):
  ...  newframe = (numpy.random.random_integers(0,255,(3,50,50)))
  ...  outv.append(newframe.astype('uint8'))
  >>> outv.close()
  >>> input = bob.io.VideoReader('testvideo.avi')
  >>> input.number_of_frames
  30
  >>> inv = input.load()
  >>> inv.shape
  (30, 3, 50, 50)
  >>> type(inv)
  <type 'numpy.ndarray'>

The loaded image files are 3D arrays (for RGB format) or 2D arrays (for greyscale) of type uint8 or uint16, while the loaded videos are 
sequences of frames, usually 4D arrays of type uint8. All the extensions and formats for images and videos supported in your version 
of |project| can be listed using the |project|'s utility `bob-config.py`.

|project| supports a number of binary formats in a manner similar to the cases shown above. Writing binary files is achieved using 
the :py:class:`bob.io.save()` with the right file extension passed as an argument, just as was shown in the example above. These additional 
formats are:
  
  * Matlab (``.mat``), Matlab arrays, supports all integer, float and complex varieties [``matlab.array.binary``];
  * bob3 (``.bindata``), supports single or double precision float numbers, only 1-D [``bob3.array.binary``];
  * bob beta (``.bin``), supports all element types in |project| and any dimensionality [``bob.array.binary``] (*deprecated*);
  * bob alpha (``.tensor``) [``tensor.array.binary``] (*deprecated*);

.. testcleanup:: *

  import shutil
  os.chdir(current_directory)
  shutil.rmtree(temp_dir)

Loading and saving Matlab data
==============================

An alternative for saving data in ``.mat`` files using :py:meth:`bob.io.save()`, would be to save them as a `HDF5`_ file which then can 
be easily read in Matlab. Similarly, instead of having to read ``.mat`` files using :py:meth:`bob.io.load()`, you can save your Matlab data 
in `HDF5`_ format, which then can be easily read from |project|. Detailed instructions about how to save and load data from Matlab to and 
from `HDF5`_ files can be found `here`__.

.. _audiosignal:

Loading and saving audio files
==============================

|project| does not yet support audio files (no wav codec). However, it is 
possible to use the `SciPy`_ module :py:mod:`scipy.io.wavfile` to do
the job. For instance, to read a wave file, just use the
:py:func:`scipy.io.wavfile.read` function.

.. code-block:: python

   >>> import scipy.io.wavfile
   >>> filename = '/home/user/sample.wav'
   >>> samplerate, data = scipy.io.wavfile.read(filename)
   >>> print type(data)
   <type 'numpy.ndarray'>
   >>> print data.shape
   (132474, 2)

In the above example, the stereo audio signal is represented as a 2D 
`NumPy` :py:class:`numpy.ndarray`. The first dimension corresponds to the
time index (132474 frames) and the second dimesnion correpsonds to one of
the audio channel (2 channels, stereo). The values in the array correpsond
to the wave magnitudes.

To save a `NumPy` :py:class:`numpy.ndarray` into a wave file, the 
:py:func:`scipy.io.wavfile.write` could be used, which also requires the
framerate to be specified.


.. include:: links.rst

.. Place here your external references

.. _ddl: http://www.hdfgroup.org/HDF5/doc/ddl.html
.. _matlab-hdf5: http://www.mathworks.ch/help/techdoc/ref/hdf5write.html
__ matlab-hdf5_
