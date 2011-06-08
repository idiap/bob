=================
 Using Databases
=================

This brief document describes how to create or use |project| Datasets from C++.

Database Design
---------------

A |project| database is composed of several key elements:

* **Arrays**: Entities that encapsulate blitz::Arrays to allow typeless
  manipulations. Arrays can be inlined or saved on external files, in one of
  |project| supported formats.
* **Arraysets**: Lists of Arrays each identified by a unique identifier. Array
  identifiers are unsigned integers starting from 1. The numbering is not
  necessarily consecutive. Arraysets have also a role attributed (e.g. these
  data are left-eye features). Arraysets maybe inlined or saved on external
  files, in one of |project| supported formats.
* **Relations**: Map arrays and arraysets forming clusters (e.g.
  target-pattern) or defining ownership of data (e.g. these data belongs to
  Bob)
* **Rules**: Defines which kinds (roles) of arrays or arraysets may be bound
  together in a Relation (e.g. Bob identity can contain on
* **Relationsets**: Lists of Relations, each identifier by a unique identifier.
  The same numbering restrictions for Arrays apply here as well. Relationsets
  also contain a set of Rules that are respected by '''all''' contained
  Relations
* **Pathlists**: Describes a set of search paths for external file lookup
* **Datasets**: Lists of Arraysets, each identified by a unique identifier. The
  same numbering restrictions for Arrays also apply here. Datasets also contain
  Relationsets that are referred by a unique name (e.g. these are
  "pattern-target" relations). Datasets may optionally contain a single
  Pathlist, composed of multiple search paths.

It is key that you understand well the relationship between these elements to
understand the |project| database API and its provided functionality.

.. todo::

  Please note this whole document too sketchy and has broken ideas. Things to
  be done:

  1. Provide real C++ examples and avoid moving the user to a website *where*
     possible.
  2. Provide a section for each type of codec if applicable, explaining
     limitations and other features.

External data formats
---------------------

|project| supports a system of underlying codecs that can import and export
external data, transparently to the end user.

You can save/load or modify Arrays in the following formats (codec names
enclosed in ``[]``): 

* Images: return RGB (uint8_t or uint16_t) 3-D arrays, or Grayscale (uint8_t or
  uint16_t) 2-D arrays as indicated [``torch.image``]. Please notice that for
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
  
  * Matlab (``.mat``), Matlab arrays, supports all integer, float and complex varieties [``matlab.array.binary``], consult MatlabCodec for more details
  * Torch3 (``.bindata``), supports single or double precision float numbers, only 1-D [``torch3.array.binary``]
  * Torch5spro beta (``.bin``), supports all element types in |project| and any dimensionality [``torch.array.binary``] (*deprecated*)
  * Torch5spro alpha (``.tensor``) [``tensor.array.binary``] (*deprecated*)
  * **HDF5** (``.hdf5``) [``hdf5.array.binary``], is the prefered format for
    enconding |project| data

You can save/load or modify Arraysets in the following formats (codec names
enclosed in ``[]``):

* Binary formats:

  * Matlab (``.mat``): several matlab arrays of the same type. (.mat version 5 -
    that is what is used by MATLAB 7.x) [``matlab.arrayset.binary``], consult
    MatlabCodec for more details
  * Torch3 (.bindata): supports arrays of single or double precision float
    numbers, only 1-D [``torch3.arrayset.binary``]
  * Torch5spro (``.bin``), supports all element types in |project| and any
    dimensionality [``torch.arrayset.binary``] (*deprecated*)
  * **HDF5** (``.hdf5``), supports all element types in |project| and any
    dimensionality [``torch.arrayset.binary``] and is **the prefered format for
    encoding |project| data**

If you want to have arrays and arraysets in external files, you have to either
know their codec names and specify that while creating the dataset or make sure
that the chosen filenames respect the extensions described before. In the
latter case, |project| will pick the appropriate codec to convert the
input/output data to/from ``blitz::Array`` s.

Loading an existing Dataset
---------------------------

Please see the documented test program (in particular, the tests named
"dbDataset_parsewrite_XYZ" in `src/cxx/database/test/dataset.cc`_).

Creating a new Dataset
----------------------

Please see the documented test program (in particular, the tests named
"dbDataset_construction", "dbDataset_arrayset", "dbDataset_relationset" at
`src/cxx/database/test/dataset.cc`_).

Transcoding (binary) files
--------------------------

Transcoding is the operation of converting files saved in one (binary) format
to another. You can transcode from/to any of the types described above, as long
as the underlying blitz::Array remains compatible with the chosen format. For
example, you can save a JPEG image as a |project| (.hdf5) file. You cannot save
a complex array inside a |project| (.hdf5) file into a Torch3 (.bindata) simply
because it only accepts single or double precision float numbers.

.. code-block:: c++

  #include "database/transcode.h"
  ...
  //array transcoding example:
  Torch::database::array_transcode("file.jpg", "file.bin");

  //arrayset transcoding example:
  Torch::database::arrayset_transcode("file.bindata", "file.bin");

|project| also provides scripts that implements the above with a few bells and
whistles. Setup |project| and then just call

.. code-block:: sh

  $ array_transcode.py from-file to-file
  # or
  $ arrayset_transcode.py from-file to-file

If you execute these scripts without any parameters, an usage instruction and a
**list of built-in codecs** will be displayed.

Extending Array and Arrayset Codecs
-----------------------------------

Please see these documented sources:

* `src/cxx/database/database/BinaryArrayCodec.h`_: BinaryArrayCodec header;
* `src/cxx/database/src/BinaryArrayCodec.cc`_: BinaryArrayCodec implementation.

You can also checkout other implementations inside the `Database package
headers`_.

.. place here your references
.. _`src/cxx/database/test/dataset.cc`: http://www.idiap.ch/software/torch5spro/browser/src/cxx/database/test/dataset.cc
.. _`src/cxx/database/database/BinaryArrayCodec.h`: http://www.idiap.ch/software/torch5spro/browser/src/cxx/database/database/BinaryArrayCodec.h
.. _`src/cxx/database/src/BinaryArrayCodec.cc`: http://www.idiap.ch/software/torch5spro/browser/src/cxx/database/src/BinaryArrayCodec.cc
.. _`Database package headers`: http://www.idiap.ch/software/torch5spro/browser/src/cxx/database/database
