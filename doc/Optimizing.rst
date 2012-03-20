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

============================================
 How to profile and optimize |project| code
============================================

Consider profiling your code **before** optimizing it. Optimization based on
guessing is pointless - **don't do this!** Here are rules you should respect
when optimizing:

1. Avoid it at all costs, optimization usually makes code unreadable and
   difficult to maintain;
2. Re-think about rule 1 for at least a day;
3. If you still need some optimization, run a full profiling session as
   explained on this guide and **understand** your bottlenecks. Optimizing
   functions that are only rarely executed and do not account for the bulk of
   the total processing time are not worth the effort. Always prefer
   readability and maintainability in these cases;
4. After completely understanding your bottlenecks, think about rule 1 again;
5. If you still decide for optimizing, then focus on the smallest set of
   routines that will get your job done. Do **no** over-optimize at the cost
   of poorer readability or maintainability.

.. note::

   Start by reading Google's `PerfTool introduction to profiling`_.

Profiling C/C++ code directly
-----------------------------

Profiling your code is a 3-step procedure: 

1. bracketing the target code with ``ProfilerStart()`` and ``ProfilerStop()``; 
2. linking against ``libprofiler.so``;
3. Analyzing the output. 

In |project|, we provide a few constructions that allow one to easy adapt any
code to support profiling. In this guide we only cover specific topics to be
addressed at |project| builds. We consider item 1 before, which is your
responsibility, has already been done and will focus this guide on getting the
code to compile/link properly. Analysis (item 3) should be conducted in the
same way as explained on `PerfTool introduction to profiling`_.

Guidelines on building against Google perftools
-----------------------------------------------

1. At your ``CMakeLists.txt``, always test to check for google-perftools
   availability:

.. code-block:: cmake

  if(googlePerfTools_FOUND)
    set(shared "${googlePerfTools_LIBRARIES}") #or make sure you link against "libprofile.so"
    add_definitions(-DHAVE_GOOGLE_PERFTOOLS)
  endif(googlePerfTools_FOUND)

2. At your code, include **optional** usage of perftools in the following
   (suggested) way:

.. code-block:: c++

  #include <cstdlib> // for std::getenv()
  #ifdef HAVE_GOOGLE_PERFTOOLS
  #include <google/profiler.h>
  #endif

  int main(void) {
    const char* profile_output = std::getenv("BOB_PROFILE");
    if (profile_output && std::strlen(profile_output)) {
  #ifdef HAVE_GOOGLE_PERFTOOLS
      std::cout << "Google perftools profile output set to " << profile_output << std::endl;
      ProfilerStart(profile_output);
  #else
      std::cout << "Google perftools were not found. Make sure they are available on your system and recompile." << std::endl;
  #endif HAVE_GOOGLE_PERFTOOLS
    }

    run_code_to_be_profiled();

  #ifdef HAVE_GOOGLE_PERFTOOLS
    if (profile_output && std::strlen(profile_output)) ProfilerStop();
  #endif
  }

With this, you tie the execution of the profiling to the setting of an
environment variable called ``BOB_PROFILE``. If you don't set the variable,
code will execute in full speed. After compilation, to run your code in profile
mode, just call your program in the following way:

.. code-block:: sh

  $ BOB_PROFILE="profile_info.out" my_program

The file ``profile_info.out`` will contain the output of the profiling session.
Use ``pprof`` as explained at Google's `PerfTool introduction to profiling`_ to
visualize the output.

Profiling |project| Python C++ extensions
-----------------------------------------

If |project| was compiled with Google perftools support, profiling python C++
extensions should be easy. Here is a recipe:

.. code-block:: sh

  #!python
  import bob

  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])

  run_code_to_be_profiled();

  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()

.. Place here your links

.. _perftool introduction to profiling: http://google-perftools.googlecode.com/svn/trunk/doc/cpuprofile.html
