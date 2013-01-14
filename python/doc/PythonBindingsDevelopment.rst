.. vim: set fileencoding=utf-8 :
.. Roy Wallace
.. 27 Mar 2012
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

=============================
 Python bindings to C++ code
=============================

Since there are two programming languages (with major differences in their
concepts) there need to be functionality to access classes and functions from
one language in the other one. To access the `C++`_ functions and classes in
python code, their needs to be **bound** to Python_.

Binding functions
~~~~~~~~~~~~~~~~~

For many practical reasons we decided to use `Boost.Python`_ to do the
bindings. `Boost.Python`_ allows to selectively expose C++ functionality to
Python_. If you plan to develop `C++`_ code for |project| that is going to be
eventually used from a Python_ environment, we strongly recommend you:

* Carefully study the `Boost Python Tutorial <http://www.boost.org/doc/libs/release/libs/python/doc/tutorial/index.html>`_, which covers basic
  information on how to bind C++ classes and methods using their clever
  templating scheme;
* Study `our own bindings <https://github.com/idiap/bob/tree/master/python>`_
  to discover design patterns we have deployed through the code and that can be
  easily re-used on your extensions.

.. include:: links.rst

.. extra links to this page go here.

.. _`Boost.Python`: http://www.boost.org/libs/python/doc
