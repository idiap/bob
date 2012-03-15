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

.. _section-usage:

=================
 Using |project|
=================

Using shipped scripts or binaries is one step away after installation. For
example:

.. code-block:: sh

  $ bob_config.py

If you installed |project| with administrator priviledges, then using bob is as
simple as importing it in your scripts. Here is an example:

.. code-block:: sh

  $ python
  ...
  >>> import bob
  >>> print bob.build.version

If you decided to either use |project| from the build location (without
properly installing it) or in case you don't have administrative priviledges on
the machine you have |project| installed, you must check a few things:

1. You must use the **same** version of python that was used to compile
   |project|. If you use the wrong version, crazy things may happen.
2. Append the location of the built or installed libraries to your
   `PYTHONPATH`, so the interpreter can find |project|.

Example:

.. code-block:: sh

  $ PYTHONPATH=<your-bob-build>/lib/python2.6 python2.6
  ...
  >>> import bob

.. note::

  You can find detailed information on how to build |project| locally at
  :doc:`BobCompilation`.
