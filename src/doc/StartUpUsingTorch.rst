=================
 Using |project|
=================

To setup a working environment at your present shell, do:

.. code-block:: sh

  $ torch-x.y/bin/shell.py

These instructions will create a clone of your current shell with the Torch
environment **appended** so your applications can find our libraries and
executables. Would you need to use the debug version of the code, add a ``-d``
(or ``--debug``) to the source command line:

.. code-block:: sh

  $ torch-x.y/bin/shell.py -d

You can use scripts like these to keep programs and setup together. Or else, to
simplify batch job submission.

Debug environments can be useful if you need to run a debugger session or to
send us a core dump with embedded debugging symbols.


Starting programs in Torch-enabled environments
-----------------------------------------------

Sometimes you just want to execute one particular program in a Torch-enabled
environment and, when you leave it, you have your previous environment back.
You can use the setup program ``shell.py`` for that purpose as well. For
example, if you want to start the python interpreter within a Torch-enabled
environment just do:

.. code-block:: sh

  $ torch-x.y/bin/shell.py -- python

When you leave the python prompt, your environment will be back to the previous
state.

Creating complete self-contained scripts
----------------------------------------

You can also create scripts that can run standalone and require no
configuration using the `Shebang`_ OS functionality. Unfortunately,
such a functionality is not standardized and is OS dependent (see `Shebang
variations`_). Here is an example of a python script that executes in a
Torch-enabled environment under *Linux*:

.. code-block:: python

  #!/WORKDIR/torch-x.y/bin/shell.py -- python
  import torch
  print torch.core.array.int16_2()

Here is another one that is just a shell script using ``bash``:

.. code-block:: sh

  #!/WORKDIR/torch-x.y/bin/shell.py --debug -- bash
  echo $TORCH_PLATFORM

.. note::

  Under BSD/MacOSX the ``/usr/bin/env`` works (as expected) breaking up the
  arguments, so should be used instead of the single shebang line showed above.
  Here is an example:

  .. code-block:: sh

    #!/usr/bin/env /WORKDIR/torch-x.y/bin/shell.py --debug -- python
    echo $TORCH_PLATFORM

C++ development
---------------

.. note::

   Be sure to familiarize yourself with `CMake`_. We use this tool for
   compiling |project|-based executables in an easy manner.

Including the |project| headers
===============================

If you need to make an include, do it in this way:

.. code-block:: c++

   #include <ip/Image.h>

Using |project| code
====================

All |project| constructions are wrapped inside the ``Torch`` namespace. To
create a |project| object, do it like this:

.. code-block:: c++

   #include <database/Video.h>
   ...
   Torch::database::Video my_video("video.mov");

Compiling the code
==================

After you have created your program, open your preferred editor creating a new
file named ``CMakeLists.txt``, that will drive your compilation. The contents
of this file should be similar to the model bellow:

.. code-block:: cmake
   
   project(example)
   cmake_minimum_required(VERSION 2.6)
   find_package(torch)
   torch_add_executable(my_example "source1.cc;source2.cc" "ip;scanning")

The line that says ``find_package(torch)`` is required. It brings in all needed
variables to compile Torch-based executables, like the location of header
files, libraries and dependencies. The following line, starting with
``torch_add_executable`` is where your code gets cooked together with Torch
code.  The first parameter defines the executable name you will find on your
prompt after compilation. The second parameter is a `CMake list`_ that contains
all source files of your program, separated by a semi-colon. The third
parameter defines the internal Torch package dependencies you need to depend
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

The line that starts with ``torch_add_executable`` is just a CMake macro that
creates a local target for CMake. The target is named after the first macro
argument. In the example above, it would be called ``my_example``. You can
extend the compilation environment and the number of linked libraries (in case
you need external dependencies) using CMake standard commands like
`include_directories`_ or `target_link_libraries`_. Example:

.. code-block:: cmake

   project(example)
   cmake_minimum_required(VERSION 2.6)
   find_package(torch)
   include_directories(/path/to/header/files/of/MyExternal1;/path/to/header/files/of/MyExternal2)
   add_definitions("-DHAS_EXTERNAL_LIB1=1" "-DHAS_EXTERNAL_LIB2")
   torch_add_executable(my_example "source1.cc;source2.cc" "ip;scanning")
   target_link_libraries(my_example "MyExternal1;MyExternal2")

Python development
------------------

Writing python code is easier than C++ because you can skip the compile-debug
loops. To be able to use Torch constructions, just call python.

.. code-block:: python

   >>> import torch

We have taken care to document all imported types using the native python help
system, so ``help()`` is your friend. Use it.

.. code-block:: python

   >>> help(torch.database.Video)

.. Place here references to all citations in lower case

.. _cmake: http://www.cmake.org
.. _include_directories: http://www.cmake.org/cmake/help/cmake-2-8-docs.html#command:include_directories
.. _target_link_libraries: http://www.cmake.org/cmake/help/cmake-2-8-docs.html#command:target_link_libraries
.. _cmake list: http://www.cmake.org/cmake/help/syntax.html 
.. _shebang: http://en.wikipedia.org/wiki/Shebang_(Unix)
.. _shebang variations: http://www.in-ulm.de/~mascheck/various/shebang/
