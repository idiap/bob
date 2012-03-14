.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
.. Wed Mar 14 12:31:35 2012 +0100
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

**************
 Introduction
**************

The following tutorials constitute a suitable starting point to get to know 
how to use the |project| library and to learn its fundamental concepts. 

They all rely on the |project| lab-like environment which is `Python`_. This
provides the user several advantages such as:

* Easy to glue all the components of an experiment within a single Python script (which does not require to be compiled)

* Scripts may easily rely on any other `Python`_ tool

As for any `Python`_ package, you first need to import |project| in your
environment before using any of its functionalities.

.. code-block:: python

  >>> import bob

In addition, you can always get information about a specific function of |project|
by using the `Python`_ help() function.

.. code-block:: python 

  >>> help(bob.machine.MLP)


.. Place here your external references

.. _python: http://www.python.org
