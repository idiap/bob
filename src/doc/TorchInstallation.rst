**************
 Installation
**************

.. _section-dependencies:

Dependencies
------------

Installing |project| Dependencies
=================================

In order to compile and develop against Torch, you must have a few tools
installed. Here is a brief description of what you need.

.. note::
   If you are at Idiap you **don't** need to install any of the packages
   mentioned on this wiki. Instead, follow the `TorchIdiapGuide`_ (special
   instructions for Idiap users).

Platforms
=========

We maintain nightly builds and unit tests that cover the following platforms:

* **Linux**: tested on Ubuntu 10.04 and it should work without problems on
  other distributions as long as you satisfy the dependencies listed bellow.
* **Mac OSX**: tested on Snow Leopard (10.6), with `MacPorts`_, but it should
  work fine on other OSX versions as well or using `Fink`_ instead.

If you find problems with a particular OS version, please `submit a new bug
report`_ so we can try to help you.

Core dependencies
=================

* `Blitz++`_: replaces the old Torch Tensor system with C++ templated
  versions that offer equivalent speed, more documentation and stability. You
  should use version 0.9 or superior. '''Please note that if you want to
  allocate very big arrays (> 2^31^ ''elements'' or 2G ''elements''), it is
  recommended that you download the latest CVS version as it includes important
  64-bit fixes'''. Also note that Blitz++ does not support 64-bit array
  indexing, so the maximum amount of positions in a single dimension is limited
  to 2^31^  (that is what fits in an signed integer), even if you decide to use
  the CVS version. This is true to the date (18/January/2011);
* `Cblas/Lapack/Atlas`_: We use these libraries for mathematical operations
  on vectors and matrices. Any modern version of this library will do;
* `Boost`_: is used for our unit test framework and python bindings. Any
  version superior to 1.34 will work;
* `Python`_: if you want to compile our python bindings we recommend using
  Python 2.5 or up;
* `NumPy`_: this dependence is used to bind blitz::Arrays to python.

Data access
===========

* `FFMpeg`_: is used to give Torch support for input and output videos.
  Torch distributions will be compiled and shipped against version 0.6 or
  superior because of licensing incompatibilities. That being said, for
  '''private development''', you can use version 0.5.1 and above. Torch will
  work seemlessly. Please note that if you decide to distribute Torch compiled
  with a version of ffmpeg prior to 0.6, then the whole package becomes GPL'ed;
* `Jpeglib`_: for JPEG support. You need version 6 at least;
* `LibXML2`_: this dependence is used for the XML Dataset implementation.
* `ImageMagick`_: is used for reading and writing image output. We
  currently compile against version 6.6.7, but other versions should work.
  Please note we make use of the C++ API (a.k.a. ImageMagick++).
* `HDF5`_: HDF5 is the format of choice for binary representation of data
  or configuration items in Torch. We currently compile against version 1.8.6,
  and version 1.8.4 might not work.
* `MatIO`_: is used for reading and writing Matlab compatible (.mat) files.
  Our nightly builds compile against version 1.34, but version 1.33 is known to
  work. Other versions should also work. Please note this dependence is
  *optional*.

Building and debugging
======================

* `CMake`_: is used to build Torch and to compile examples. You need at
  least version 2.8;
* `Google Perftools`_: if you want to compile profiling extensions. We have
  used version 1.6, but version 1.5 will do the work as well. Please note that
  the use of this package is optional.
* `Sphinx`_: is used to generate the user manuals and python API reference
  guide. We use the latest available version of Sphinx, but earlier versions
  should work.
* `Doxygen`_: is used for extracting C/C++ documentation strings from code
  and building a system of webpages describing the C/C++ Torch API.

.. note::
   If your OS cannot satisfy the minimal required versions of the packages, you
   may have to install and compile some or all of the dependencies above in a
   private (prefix) directory. If you choose to do so, you must instruct cmake
   to look for libraries and header files first on your newly created prefix by
   setting the environment variable CMAKE_PREFIX_PATH to point to that prefix
   like this:

   .. code-block:: sh

      $ export CMAKE_PREFIX_PATH=/path/to/the/root/of/your/packages

Extra packages we recommend
===========================

These are packages that are *not* required to compile or run torch examples,
but make a nice complement to the installation and provides you with the
ability to plot and interact with Torch:

* `Scipy`_: A set of scientific-related python-based utilities
* `Matplotlib`_: A matlab-like python plotting environment
* `IPython`_: A powerful replacement for your python shell that provides bells
  and whistles
* `H5py`_ and `Tables`_: HDF5 bindings to Python

Notes for specific platforms
----------------------------

Ubuntu
======

A single command line that will install all required packages under Ubuntu
(tested on Ubuntu 10.04 LTS):

.. code-block:: sh

   $ sudo apt-get install cmake libatlas-base-dev libblitz0-dev libgoogle-perftools-dev ffmpeg libavcodec-dev libswscale-dev libboost-all-dev libavformat-dev libjpeg-dev graphviz libxml2-dev libmatio-dev libmagick++9-dev python-scipy python-numpy python-matplotlib h5utils hdf5-tools libhdf5-doc python-h5py python-tables python-tables-doc libhdf5-serial-1.8.4 libhdf5-serial-dev

Mac OSX
=======

This is a recipe for compiling Torch under your Mac OSX using Snow Leopard. It
should be possible, but remains untested, to execute similar steps under OSX
Leopard (10.5.X). We would like to hear if you have a success story or problems
`submit a new bug report`_.

This recipe assumes you have already gone through the standard,
well-documented, `MacPorts installation instructions`_ and has a prompt just in
front of you and a checkout of torch you want to try out. Then, just do, at
your shell prompt:

.. code-block:: sh

   $ sudo port install cmake blitz ffmpeg jpeg atlas python26 python_select gcc44 gcc_select py26-numpy matio imagemagick py26-ipython py26-matplotlib google-perftools doxygen py26-sphinx hdf5-18 py26-h5py py26-tables boost +python26
   $ # go for a long coffee

You can install also git if you want to submit patches to us:

.. code-block:: sh

   $ sudo port install  git-core +python26

For compiling Torch under OSX, we recommend the use of "llvm-gcc" instead of
plain gcc. After running the command above, do the following:

.. code-block:: sh

   $ sudo gcc_select llvm-gcc42
   #or
   $ sudo gcc_select mp-llvm-gcc42

We also have fortran files that need compilation. Make sure ``gfortran`` is
accessible from the command line before trying to compile. Specifically, the
MacPorts installation may not put ``gfortran`` on the command line and call the
executable in a different way (in my system it is called ``gfortran-mp-4.4``).
To make cmake find the fortran compiler you will have to create, manually, a
symbolic link from this binary. Here are the instructions:

.. code-block:: sh

   $ cd /opt/local/bin; sudo ln -s gfortran-mp-4.4 gfortran

.. warning::
   * Torch/Blitz python bindings will not compile in _release_ mode with plain
     gcc-4.2 (blitz causes a segmentation fault at the compiler). This is why
     we recommend to use the llvm gcc bridge instead.

After you have gone through these installation steps, you can proceed with the
normal TorchCompilation instructions. If you have followed the
`MacPorts`_ installation guide to the letter, your environment should be
correctly set. You **don't** need to setup any other environment variable.

Obtaining the code
------------------

To install Torch you need first to set your mind on what to install. You can
choose between a released stable version from :doc:`TorchDistribution` or
checkout and build yourself following :ref:`section-compilation`.

.. warning::
  *Make sure to read  and install all requirements defined in*
  :ref:`section-dependencies`, *prior to running Torch applications.*

Grab a tarball and change into the directory of your choice, let's say
``WORKDIR``:

.. code-block:: sh

  $ cd WORKDIR
  $ wget |torchweb|/nightlies/torch-nightly-latest.tar.gz
  $ tar xvfz torch-nightly-latest.tar.gz

.. _section-checkout:

Checking out |project|
----------------------

To checkout you currently need access to Idiap's internal filesystem (to be
open-sourced soon!):

.. code-block:: sh

   $ git clone username@machine.idiap.ch:/idiap/group/torch5spro/git/torch5spro.git

You have to fill the ``username`` and ``machine`` bits with your Idiap username
and the machine you want to use for ssh. Please note that in order to push
changes you need that ``machine`` does have `BuildBot`_ packages installed so
that our build server is correctly informed of changes. Please contact one of
the |project| developers to learn about existing machines with packages
pre-installed.

.. _section-compilation:

Compiling the code
------------------

If you decided to download a source-form distribution. You need to compile it
in the destination machine before using it. Just execute:

.. code-block:: sh
   
   $ cd torch5spro-x.y
   $ bin/debug.sh
   # or
   $ bin/release.sh

This will compile and install (under the directory `install` in the current
working directory) all libraries, executables and headers available in
|project|. You can fine tune the behavior of these shell scripts by looking up
its help message:

.. code-block:: sh

   $ bin/debug.sh --help
   # or
   $ bin/release.sh --help

Troubleshooting compilation
===========================

Most of the problems concerning compilation come from not satisfying correctly
the :ref:`section-dependencies` (such as `FFmpeg`_, `ImageMagick`_, etc). Start
by double-checking every dependency or base OS and check everything is as
expected. If you cannot go through, please `submit a new bug report`_ in
our tracking system. At this time make sure to specify your OS version and the
versions of the external dependencies so we can try to reproduce the failure.


.. Place here references to all citations in lower case

.. _macports: http://www.macports.org
.. _macports installation instructions: http://www.macports.org/install.php
.. _fink: http://www.finkproject.org
.. _submit a new bug report: https://www.idiap.ch/software/torch5spro/newticket
.. _blitz++: http://www.oonumerics.org/blitz
.. _cmake: http://www.cmake.org
.. _ffmpeg: http://www.ffmpeg.org
.. _jpeglib: http://www.ijg.org
.. _cblas/lapack/atlas: http://www.netlib.org Cblas/Lapack/Atlas
.. _boost: http://www.boost.org
.. _python: http://www.python.org
.. _google perftools: http://code.google.com/p/google-perftools
.. _numpy: http://http://numpy.scipy.org
.. _libxml2: http://xmlsoft.org
.. _doxygen: http://www.doxygen.org
.. _sphinx: http://sphinx.pocoo.org
.. _matio: http://matio.sourceforge.net
.. _imagemagick: http://www.imagemagick.org
.. _hdf5: http://www.hdfgroup.org/HDF5
.. _scipy: http://www.scipy.org
.. _ipython: http://ipython.scipy.org
.. _h5py: http://code.google.com/p/h5py/
.. _tables: http://www.pytables.org
.. _matplotlib: http://matplotlib.sourceforge.net
.. _torchidiapguide: https://www.idiap.ch/software/torch5spro/wiki/TorchIdiapGuide
.. _buildbot: http://http://trac.buildbot.net
