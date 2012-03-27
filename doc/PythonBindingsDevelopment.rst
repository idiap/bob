.. vim: set fileencoding=utf-8 :
.. Roy Wallace
.. 27 Mar 2012
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

=============================
 Python bindings to C++ code
=============================

.. todo:: 

  Describe the concepts of python bindings and how to write them, with an example

Guidelines
----------

* When you bind C++ methods to python, use `Boost.Python`_. Try to follow the 
  examples in already existing bindings.
* Avoid allocating memory in C++ and exporting it to python. Instead, allocate 
  the memory in python and give it as reference parameters to your functions.
  Allocating blitz::Arrays in C++ and export them to python is possible, but 
  error prone. 
* If you want your bound classes to allocate memory, please write python 
  functions and add it to your bound classes. Try to follow the examples in the
  code.

.. include:: links.rst
.. _`Boost.Python`: http://www.boost.org/libs/python/doc

