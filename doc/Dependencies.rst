.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Tue 27 Mar 2012 10:58:49 CEST 
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

.. _section-dependencies:

==============
 Dependencies
==============

This is a listing of current |project| dependencies showing minimum required
versions where relevant and the respective project licenses. When a project is
covered by multiple licenses, we show which we are presently using (using
**bold** face). The following sections have detailed information on why the
dependence is required for the project and installation instructions for
specific platforms.

Required at Runtime
===================

+----------------------+--------------+---------------------------------------+
| Package Name         | Min. Version | License                               |
+======================+==============+=======================================+
| Std. C/C++ Libraries | any          | Depends on the compiler               |
+----------------------+--------------+---------------------------------------+
| `Blitz++`_           | 0.9          | `Artistic-2.0`_ or LGPLv3+ or GPLv3+  |
+----------------------+--------------+---------------------------------------+
| `Lapack`_            | any          | BSD-style                             |
+----------------------+--------------+---------------------------------------+
| `Python`_            | 2.5          | `Python-2.0`_                         |
+----------------------+--------------+---------------------------------------+
| `Boost`_             | 1.34         | `BSL-1.0`_                            |
+----------------------+--------------+---------------------------------------+
| `NumPy`_             | 1.3          | `BSD-3-Clause`_                       |
+----------------------+--------------+---------------------------------------+
| `Scipy`_             | 0.7?         | `BSD-3-Clause`_                       |
+----------------------+--------------+---------------------------------------+
| `Matplotlib`_        | 0.99         | Based on `Python-2.0`_                |
+----------------------+--------------+---------------------------------------+
| `fftw`_              | 3.0?         | `GPL-2.0`_ or later (also commercial) |
+----------------------+--------------+---------------------------------------+
| `SQLAlchemy`_        | 0.5          | `MIT`_                                |
+----------------------+--------------+---------------------------------------+
| `ImageMagick`_       | 6.5          | `Apache-2.0`_                         |
+----------------------+--------------+---------------------------------------+
| `HDF5`_              | 1.8.4        | `HDF5 License`_ (BSD-like, 5 clauses) |
+----------------------+--------------+---------------------------------------+
| `argparse`_          | 1.2          | `Python-2.0`_                         |
+----------------------+--------------+---------------------------------------+

Strongly Recommended Add-Ons
============================

+----------------------+--------------+----------------------------------------------+
| Package Name         | Min. Version | License                                      |
+======================+==============+==============================================+
| `FFMpeg`_            | 0.5          | `LGPL-2.1`_ or later, or `GPL-2.0`_ or later |
+----------------------+--------------+----------------------------------------------+
| `MatIO`_             | 1.3.3?       | `BSD-2-Clause`_                              |
+----------------------+--------------+----------------------------------------------+
| `Qt4`_               | 4.7?         | `LGPL-2.1`_ (also commercial)                |
+----------------------+--------------+----------------------------------------------+
| `VLFeat`_            | 0.9.14       | `BSD-2-Clause`_                              |
+----------------------+--------------+----------------------------------------------+
| `OpenCV`_            | 2.1?         | `BSD-3-Clause`_                              |
+----------------------+--------------+----------------------------------------------+
| `LIBSVM`_            | 2.89+        | `BSD-3-Clause`_                              |
+----------------------+--------------+----------------------------------------------+

Build Dependencies
==================

+----------------------+--------------+------------------+
| Package Name         | Min. Version | License          |
+======================+==============+==================+
| `Git`_               | 1.6?         | `GPL-2.0`_       |
+----------------------+--------------+------------------+
| `CMake`_             | 2.8          | `BSD-3-Clause`_  |
+----------------------+--------------+------------------+
| `Google Perftools`_  | 0.8?         | `BSD-3-Clause`_  |
+----------------------+--------------+------------------+
| `Sphinx`_            | 0.6          | `BSD-2-Clause`_  |
+----------------------+--------------+------------------+
| `Doxygen`_           | 1.6?         | `GPL-2.0`_       |
+----------------------+--------------+------------------+
| `Dvipng`_            | 1.12?        | `GPL-3.0`_       |
+----------------------+--------------+------------------+

Recommended Software
====================

+----------------------+--------------+------------------+
| Package Name         | Min. Version | License          |
+======================+==============+==================+
| `IPython`_           | any          | `BSD-3-Clause`_  |
+----------------------+--------------+------------------+

Description of |project| Dependencies
-------------------------------------

In order to compile and develop against bob, you must have a few tools
installed. Here is a brief description of what you need.

.. note::
   If you are at Idiap you **don't** need to install any of the packages
   mentioned on this wiki. Instead, follow `Bob's Idiap Guide`_ (special
   instructions for Idiap users).

Platforms
=========

We maintain nightly builds and unit tests that cover the following platforms:

* **Linux**: tested on Ubuntu 10.04/12.04 and it should work without problems
  on other distributions as long as you satisfy the dependencies listed on this
  section;
* **Mac OSX**: tested on Snow Leopard (10.6), with `MacPorts`_, but it should
  work fine on other OSX versions as well or using `Fink`_ instead as long as
  you install all required dependencies.

If you find problems with a particular OS version, please `submit a new bug
report`_ so we can try to help you.

Core dependencies
=================

* `Blitz++`_: replaces the old bob Tensor system with C++ templated
  versions that offer equivalent speed, more documentation and stability. You
  should use version 0.9 or superior. '''Please note that if you want to
  allocate very big arrays (> 2^31^ ''elements'' or 2G ''elements''), it is
  recommended that you download the latest CVS version as it includes important
  64-bit fixes'''. Also note that Blitz++ does not support 64-bit array
  indexing, so the maximum amount of positions in a single dimension is limited
  to 2^31^  (that is what fits in an signed integer), even if you decide to use
  the CVS version. This is true to the date (18/January/2011);
* `Lapack`_: We use this library for mathematical operations
  on vectors and matrices. Any modern version of this library will do;
* `FFTW`_: Provides fast computation of Fourier, Sine and Cosine transforms
* `Boost`_: is used for our unit test framework and python bindings. Any
  version superior to 1.34 will work;
* `Python`_: if you want to compile our python bindings we recommend using
  Python 2.5 or up;
* `NumPy`_: this dependence is used to bridge blitz::Arrays to python.
* `Scipy`_: A set of scientific-related python-based utilities
* `Matplotlib`_: A `MATLAB`_-like python plotting environment
* `Qt4`_: This library is used as the basis for the face detection and
  localization framework (Visioner). This dependence is *optional*. Face
  localization and detection will only be compiled if you have that installed.
* `argparse`_: is used for argument parsing in some of our python utilities. If
  you are not compiling against Python 2.7 or superior, you need to install it
  separately.

Data access
===========

* `SqlAlchemy`_: A Python SQL toolkit and Object Relational Mapper. This is
  used for storing and querying more complex databases. You need at least
  version 0.5.
* `FFMpeg`_: is used to give bob support for input and output videos.
  bob distributions will be compiled and shipped against version 0.6 or
  superior because of licensing incompatibilities. That being said, for
  '''private development''', you can use version 0.5.1 and above. bob will
  work seemlessly. Please note that if you decide to distribute bob compiled
  with a version of ffmpeg prior to 0.6, then the whole package becomes GPL'ed;
* `ImageMagick`_: is used for reading and writing image output. We
  currently compile against version 6.6.7, but other versions should work.
  Please note we make use of the C++ API (a.k.a. ImageMagick++).
* `HDF5`_: HDF5 is the format of choice for binary representation of data
  or configuration items in bob. We currently compile against version 1.8.6,
  Version 1.8.4 (and before) might not work.
* `MatIO`_: is used for reading and writing `MATLAB`_ compatible (.mat) files.
  Our nightly builds compile against version 1.34, but version 1.33 is known to
  work. Other versions should also work. Please note this dependence is
  *optional*.
* `VLFeat`_: is used for calculating SIFT features. This is an *optional*
  dependencies. If you have VLFeat installed, additional C++ and Python
  bindings will be compiled and integrated to |project|.

.. _basic-build:

Building and debugging
======================

These are packages you need for compiling |project|, but do not depend at
during runtime.

* `Git`_: is used as our version control system. You need it if you want to
  perform a fresh checkout of sources beforem compiling;
* `CMake`_: is used to build bob and to compile examples. You need at
  least version 2.8;
* `Google Perftools`_: if you want to compile profiling extensions. We have
  used version 1.6, but version 1.5 will do the work as well. Please note that
  the use of this package is optional.
* `Sphinx`_: is used to generate the user manuals and python API reference
  guide. We use the latest available version of Sphinx, but earlier versions
  should work.
* `Doxygen`_: is used for extracting C/C++ documentation strings from code
  and building a system of webpages describing the C/C++ bob API.
* `Dvipng`_: is required for LaTeX-like code conversion to HTML. Not having it
  will cause equations to be displayed using LaTeX-code instead of being nicely
  formatted.

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

These are packages that are *not* required to compile or run bob examples,
but make a nice complement to the installation and provides you with the
ability to plot and interact with bob:

* `IPython`_: A powerful replacement for your python shell that provides bells
  and whistles
* `LIBSVM`_: A software framework for support vector classification
* `OpenCV`_: A Computer Vision library which is currently used by the 
  |project| Data Acquisition module.

Notes for specific platforms
----------------------------

Ubuntu 10.04 (LTS)
==================

A single command line that will install all required packages under Ubuntu
(tested on Ubuntu 10.04 LTS):

.. code-block:: sh

   $ sudo apt-get install git-core cmake liblapack-dev libatlas-base-dev libblitz0-dev libgoogle-perftools0 ffmpeg libavcodec-dev libswscale-dev libboost-all-dev libavformat-dev graphviz libmatio-dev libmagick++9-dev python-scipy python-numpy python-matplotlib ipython h5utils hdf5-tools libhdf5-doc libhdf5-serial-dev python-argparse python-sqlalchemy python-sphinx dvipng libqt4-dev libfftw3-dev libcv-dev libhighgui-dev libcvaux-dev libsvm-dev doxygen python-sphinx texlive-fonts-recommended

.. note::

  Support for libgoogle-perftools-dev in Ubuntu 10.04 LTS is broken so you will
  not be able to profile |project| using this support. You may still use
  Valgrind or other profiling tools of your choice.

.. note::

  You will not find a pre-packaged version of VLfeat (SIFT feature extraction)
  on Ubuntu distributions by default. You can still add the PPA by following
  instructions on the `VLfeat launchpad webpage`_.

Ubuntu 11.10
============

A single command line that will install all required packages under Ubuntu
(tested on Ubuntu 11.10):

.. code-block:: sh

   $ sudo apt-get install git-core cmake liblapack-dev libatlas-base-dev libblitz0-dev libgoogle-perftools-dev ffmpeg libavcodec-dev libswscale-dev libboost-all-dev libavformat-dev graphviz libmatio-dev libmagick++9-dev python-scipy python-numpy python-matplotlib ipython h5utils hdf5-tools libhdf5-doc libhdf5-serial-dev python-sqlalchemy python-sphinx dvipng libqt4-dev libfftw3-dev libcv-dev libhighgui-dev libcvaux-dev libsvm-dev doxygen python-sphinx texlive-fonts-recommended

.. note::

  You will not find a pre-packaged version of VLfeat (SIFT feature extraction)
  on Ubuntu distributions by default. You can still add the PPA by following
  instructions on the `VLfeat launchpad webpage`_.

Ubuntu 12.04 (LTS)
==================

A single command line that will install all required packages under Ubuntu
(tested on Ubuntu 12.04):

.. code-block:: sh

   $ sudo apt-get install git-core cmake liblapack-dev libatlas-base-dev libblitz0-dev libgoogle-perftools-dev ffmpeg libavcodec-dev libswscale-dev libboost-all-dev libavformat-dev graphviz libmatio-dev libmagick++-dev python-scipy python-numpy python-matplotlib ipython h5utils hdf5-tools libhdf5-doc libhdf5-serial-dev python-sqlalchemy python-sphinx dvipng libqt4-dev libfftw3-dev libcv-dev libhighgui-dev libcvaux-dev libsvm-dev doxygen python-sphinx texlive-fonts-recommended

.. note::

  You will not find a pre-packaged version of VLfeat (SIFT feature extraction)
  on Ubuntu distributions by default. You can still add the PPA by following
  instructions on the `VLfeat launchpad webpage`_.

Mac OSX
=======

This is a recipe for installing |project| dependencies under your Mac OSX using
Snow Leopard (10.6) or Lion (10.7). It should be possible, but remains
untested, to execute similar steps under OSX Leopard (10.5.X). We would like to
hear from you! If you have a success story or problems `submit a new bug
report`_.

This recipe assumes you have already gone through the standard,
well-documented, `MacPorts installation instructions`_ and has a prompt just in
front of you and a checkout of bob you want to try out. Then, just do, at your
shell prompt:

.. code-block:: sh

   $ sudo port install cmake blitz ffmpeg python26 python_select gcc44 gcc_select py26-numpy -atlas matio imagemagick py26-ipython py26-matplotlib google-perftools doxygen py26-sphinx texlive-bin hdf5-18 py26-argparse qt4-mac boost +python26 python26-scipy +no_atlas fftw-3 vlfeat opencv +python26 +qt4 libsvm +python26 +tools dvipng
   $ # go for a long coffee 

After the installation has finished, make sure you select python 2.6 (macports)
as your default shell:

.. code-block:: sh

  $ sudo port select python python26

This will make sure you use the correct version of python by default, but it is
not strictly necessary, if you remember choosing it correctly when starting a
prompt manually.

.. note::

  If you are installing on a machine running OSX Lion (10.7), use qt4-mac-devel
  (version 4.8) instead of the package "qt4-mac".

.. note::

  This setup will guide you to choose Python_ 2.6 as the interpreter where
  |project| will run. You can use either Python_ 2.5 or Python_ 2.7 as well.
  Make the required modifications on the instructions above so to install
  packages for that version of python instead. 

You can also install git if you want to submit patches to us:

.. code-block:: sh

   $ sudo port install  git-core +python26

For compiling |project| under OSX, we recommend the use of "llvm-gcc" instead
of plain gcc. After running the command above, do the following:

.. code-block:: sh

   $ sudo port select gcc llvm-gcc42
   #or
   $ sudo port select gcc mp-llvm-gcc42

.. warning::

  If you have an old ports tree, you may have to do instead:

  .. code-block:: sh

     $ sudo gcc_select llvm-gcc42
     #or
     $ sudo gcc_select mp-llvm-gcc42

.. warning::
   * The current MacPorts versionf blitz does not compile with anything newer
     than gcc-4.2.

If you have followed the `MacPorts`_ installation guide to the letter, your
environment should be correctly set. You **don't** need to setup any other
environment variable.

.. include:: links.rst
