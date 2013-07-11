.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed Jan 11 14:43:35 2012 +0100
..
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

============================================
 How to Profile and Optimize |project| Code
============================================

Consider profiling your code **before** optimizing it. Optimization based on
guessing is pointless - **don't do this!** Here are rules you should respect
when optimizing.

1. Avoid it at all costs, optimization usually makes code unreadable and
   difficult to maintain.
2. Re-think about rule 1 for at least a day.
3. If you still need some optimization, run a full profiling session as
   explained on this guide and **understand** your bottlenecks. Optimizing
   functions that are only rarely executed and do not account for the bulk of
   the total processing time are not worth the effort. Always prefer
   readability and maintainability in these cases.
4. After completely understanding your bottlenecks, think about rule 1 again.
5. If you still decide for optimizing, then focus on the smallest set of
   routines that will get your job done. Do **not** over-optimize at the cost
   of poorer readability or maintainability.

.. note::

   Start by reading Google's `PerfTool introduction to profiling`_.

Profiling C/C++ code directly
-----------------------------

Profiling your code is a 3-step procedure.

1. Bracket the target code with ``ProfilerStart()`` and ``ProfilerStop()``.
2. Link against ``libprofiler.so``.
3. Analyse the output.

In |project|, we provide a few constructions that allow you to easily adapt any
code to support profiling. In this guide we only cover specific topics to be
addressed for |project| builds. We consider that item 1, which is your
responsibility, has already been done and will focus this guide on getting the
code to compile/link properly (item 2). Analysis (item 3) should be conducted
in the same way as explained on `PerfTool introduction to profiling`_.

Guidelines on building against Google perftools
-----------------------------------------------

1. While compiling |project|, make sure you have added support for Google
   Perftools, during its configuration, by using the flag ``WITH_PERFTOOLS``.
   For example:

   .. code-block:: cmake

     $ cmake -DWITH_PERFTOOLS=ON -DCMAKE_BUILD_TYPE=Debug ..

   Before doing so, make sure you have followed your operating system
   instructions to install Google Perftools and that cmake can find it.

2. Within your C++ code, include usage of perftools in the following
   (suggested) way:

.. code-block:: c++

  #include <google/profiler.h>

  int main(void) {
    ProfilerStart("/tmp/profile_info.out");

    run_code_to_be_profiled();

    ProfilerStop();
  }

The file ``/tmp/profile_info.out`` will contain the output of the profiling
session.  Use ``pprof`` as explained at Google's `PerfTool introduction to
profiling`_ to visualize the output.

Profiling |project| Python C++ extensions
-----------------------------------------

If |project| was compiled with Google perftools support like indicated above,
profiling python C++ extensions should be easy. Here is a recipe:

.. code-block:: python

  import bob

  bob.core.ProfilerStart("/tmp/profile_info.out")

  run_code_to_be_profiled()

  bob.core.ProfilerStop()

.. Place here your links

.. _perftool introduction to profiling: http://google-perftools.googlecode.com/svn/trunk/doc/cpuprofile.html
