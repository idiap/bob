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

=================
 C++ development
=================

.. note::

   Be sure to familiarize yourself with `CMake`_. We use this tool for
   compiling |project|-based executables in an easy manner.

Including the |project| headers
===============================

If you need to make an include, do it in this way:

.. code-block:: c++

   #include <io/Video.h>

Using |project| code
====================

All |project| constructions are wrapped inside the ``bob`` namespace. To
create a |project| object, do it like this:

.. code-block:: c++

   #include <io/Video.h>
   ...
   bob::io::Video my_video("video.mov");

Compiling the code
==================

After you have created your program, open your preferred editor creating a new
file named ``CMakeLists.txt``, that will drive your compilation. The contents
of this file should be similar to the model bellow:

.. code-block:: cmake
   
   project(example)
   cmake_minimum_required(VERSION 2.6)
   find_package(bob)
   bob_add_executable(my_example "source1.cc;source2.cc" "ip;scanning")

The line that says ``find_package(bob)`` is required. It brings in all needed
variables to compile bob-based executables, like the location of header
files, libraries and dependencies. The following line, starting with
``bob_add_executable`` is where your code gets cooked together with |project|
code.  The first parameter defines the executable name you will find on your
prompt after compilation. The second parameter is a `CMake list`_ that contains
all source files of your program, separated by a semi-colon. The third
parameter defines the internal |project| package dependencies you need to depend
on, also separated by semi-colons. It is advisable to only introduce a
*minimal* set of dependencies you need to compile and link a program.  Bringing
in more dependencies than you need introduce unexpected behavior.

After generating the adequate ``CMakeLists.txt`` file for your project, all it
remains is to compile the code. You can do this with these simple steps on your
prompt:

.. code-block:: sh
   
   $ cmake .
   $ make

.. note::

   After running cmake, a ``CMakeCache.txt`` file will be produced. If you
   experience any problems with running cmake, it is recommended to first
   remove this cache file and try again.

Special case: Introducing external header files and libraries
=============================================================

The line that starts with ``bob_add_executable`` is just a CMake macro that
creates a local target for CMake. The target is named after the first macro
argument. In the example above, it would be called ``my_example``. You can
extend the compilation environment and the number of linked libraries (in case
you need external dependencies) using CMake standard commands like
`include_directories`_ or `target_link_libraries`_. Example:

.. code-block:: cmake

   project(example)
   cmake_minimum_required(VERSION 2.6)
   find_package(bob)
   include_directories(/path/to/header/files/of/MyExternal1;/path/to/header/files/of/MyExternal2)
   add_definitions("-DHAS_EXTERNAL_LIB1=1" "-DHAS_EXTERNAL_LIB2")
   bob_add_executable(my_example "source1.cc;source2.cc" "ip;scanning")
   target_link_libraries(my_example "MyExternal1;MyExternal2")

.. Place here references to all citations in lower case

.. _cmake: http://www.cmake.org
.. _include_directories: http://www.cmake.org/cmake/help/cmake-2-8-docs.html#command:include_directories
.. _target_link_libraries: http://www.cmake.org/cmake/help/cmake-2-8-docs.html#command:target_link_libraries
.. _cmake list: http://www.cmake.org/cmake/help/syntax.html 
.. _shebang: http://en.wikipedia.org/wiki/Shebang_(Unix)
.. _shebang variations: http://www.in-ulm.de/~mascheck/various/shebang/
