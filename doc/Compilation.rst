.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed Jan 11 14:43:35 2012 +0100
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

.. _section-compilation:

==============================
 Compiling |project| Yourself
==============================

Obtaining the code
------------------

To compile |project| locally you need to first set your mind on what to
work with. You can choose between a released stable version from `Bob's website`
or checkout and build yourself as explained below.

.. warning::

  Make sure you read and install all requirements defined in
  :doc:`Dependencies`, prior to compiling and using |project|.

Grab a tarball and change into the directory of your choice, let's say
``mybob``:

.. code-block:: sh

  $ mkdir mybob
  $ cd mybob
  # getting version 1.0 as a tar ball
  $ wget https://github.com/idiap/bob/tarball/v1.0
  $ tar xvfz idiap-bob-v1.0-0-gxyzabc.tar.gz
  $ cd idiap-bob-xyzabc

.. _section-checkout:

Cloning |project|
-----------------

To checkout |project|, do the following at your shell prompt:

.. code-block:: sh

   $ git clone https://github.com/idiap/bob

Compiling the code
------------------

Just execute:

.. code-block:: sh
   
   $ cd bob
   $ mkdir build
   $ cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release ..
   # download databases for the build: requires an internet connection!
   $ make db-download
   # run the whole build
   $ make

After building, you can execute all unit tests to make sure the build is (as)
reliable (as it can be):

.. code-block:: sh

  $ make test
  # if you prefer, you can directly use ctest
  $ ctest -V

The documentation can be generated with other specific make targets:

.. code-block:: sh

   $ make doxygen #generates the C++ API documentation
   $ make sphinx #generates the user guide

You don't need to to install |project| to use it. If you wish to do it though,
you need to do it by calling ``make`` again:

.. code-block:: sh

  $ make install
  $ make install-doxygen
  $ make install-sphinx

The installation base directory is set to cmake's default, which is usually on
an administrator restricted area, such as ``/usr/local``. If you wish to install
the build in a different directory, you need to tell ``cmake`` the installation
prefix:

.. code-block:: sh

  # installs on ../install
  $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install ..
  $ make
  $ make install

Troubleshooting compilation
===========================

Most of the problems concerning compilation come from not satisfying correctly
the :ref:`section-dependencies`. Start by double-checking every dependency or
base OS and check everything is as expected. If after exhausting all of these 
possibilities you are still unable to compile |project|, please `submit a new bug report`_ 
in our tracking system. At this time make sure to specify your OS version and the versions 
of the external dependencies so we can try to reproduce the failure.

.. include:: links.rst
