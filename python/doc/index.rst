.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Laurent El Shafey <laurent.el-shafey@idiap.ch>
.. Sun Apr 3 19:18:37 2011 +0200
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

============
 Python API
============

|project| is written in a mix of Python and C++ and is designed to be both 
efficient and to reduce development time. Most of the |project| C++ 
functionality is bound to Python using `Boost Python`_. This provides the
users several advantages:

* Possibility to call efficient C++ functions from the Python interpreter
* Easy to glue all the components of an experiment within a single Python
  script (which does not require to be compiled)
* Scripts may easily rely on any other Python tool

This section describes the features of |project| following its organization
in packages. It includes both a user guide and a reference manual.

We recall that to use |project| from Python, you just need to add the path to
the built library to the PYTHONPATH environment variable before launching a 
Python interpreter and importing bob. This has already been described in the
installation section.

.. code-block:: sh

  $ PYTHONPATH=YOUR_PATH_TO_BOB python
  ...
  >>> import bob

Finally, you can always get information about a specific function by using the
Python help() function:

.. code-block:: sh

  >>> help(bob.machine.MLP)


.. toctree::
   :maxdepth: 2

   ../core/doc/index
   ../io/doc/index
   ../math/doc/index
   ../measure/doc/index
   ../sp/doc/index
   ../ip/doc/index
   ../db/doc/index
   ../machine/doc/index
   ../trainer/doc/index
   ../visioner/doc/index
   ../daq/doc/index

.. References

.. _boost python: http://www.boost.org/doc/libs/release/libs/python/doc/index.html
