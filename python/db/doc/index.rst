.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Thu Jun 23 11:55:24 2011 +0200
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

.. Index file for the bob databases

===========
 Databases
===========

This package describes the database API for |project|. Database APIs establish
how your programs can query for file lists using known pre-coded protocols that
assure reproducibility. This package contains only the base API for you to
create and distribute new databases and a single, very simple example using the
publicly available Iris Flower Dataset.

Build a database package for Bob goes pretty much like building a `satellite
package <https://github.com/idiap/bob/wiki/Satellite-Packages>`_. For examples
and details, have a look at our `satellite package portal
<https://github.com/idiap/bob/wiki/Satellite-Packages>`_.

.. automodule:: bob.db

General Utilities
=================

.. automodule:: bob.db.utils

Central Database Manager API
============================

.. automodule:: bob.db.driver

Iris Flower Dataset
===================

.. automodule:: bob.db.iris
