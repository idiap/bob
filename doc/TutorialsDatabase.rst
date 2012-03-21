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

**********
 Database
**********

|project| provides an API to easily query and interface with well know 
biometric databases. A database contains information about the organization
of the files, functions to query information such as the data which might be
used for training a model, but it does **not** contain the data itself.
Most of the databases are stored in a `sqlite`_ file, whereas the smallest 
ones are stored as filelists.

.. testsetup:: *

   import bob


MOBIO database
==============

Let's consider an example with the freely available `MOBIO`_ database, which
consists in bi-modal (audio and video) data from 152 people. For this example,
we will use the still images of the database. The database is created as any 
other `Python` object.

.. doctest::

   >>> mobioDb = bob.db.mobio.Database()

If the `sqlite`_ file containing the `MOBIO`_ database has not been created 
during the |project| build, the previous command will return a runtime error.
Fortunately, |project| provides a binary utility called `dbmanage.py` which 
allows to perform basic operations on a database, such as creating it, dumping
the files, etc. In case of problem, running the following command will 
generate the database.

.. code-block:: sh

   $ dbmanage.py mobio create

Once the database has been created, it is possible to query information about
it. For instance, to retrieve the list of clients, the 
:py:meth:`bob.db.mobio.Database.clients()` could be called.

.. doctest::

   >>> clientsList = mobioDb.clients()

The |project| database also contains information about the protocols. In 
particular, the :py:meth:`bob.db.mobio.Database.clients()` can be 
parametrised to only return the list of identities which can be used to train
a model (`world` subset) for a specified protocol.

.. doctest::

   >>> clientsSublist = mobioDb.clients(protocol = 'male', groups = 'world')
   >>> clientsSublistSorted = sorted(clientsSublist)
   >>> print clientsSublistSorted[0]
   202
   
Then, if we would like to retrieve the list of files associated with this 
identity, the :py:meth:`bob.db.mobio.Database.files()` method will be of 
help.

.. doctest::

   >>> fileList0 = mobioDb.files(protocol = 'male', groups = 'world', model_ids = (clientsSublistSorted[0],))
   >>> print len(fileList0)
   192

In the previous case, the returned list of filenames contains relative path 
without extension. However, it is posssible to provides a base directory and
an extension argument to the function, that will respectively prepend and 
append them to the list of filenames.

.. doctest::

   >>> fileList1 = mobioDb.files(protocol = 'male', groups = 'world', model_ids = (clientsSublistSorted[0],), directory = '/MYDIR', extension = '.pgm')
   >>> print fileList1 # doctest: +SKIP

Finally, it is possible to check that a given base directory contains all the
files of a database using the `checkfiles` command of the `dbmanage.py` script.

.. code-block:: sh

   $ dbmanage.py mobio checkfiles -d /MYDIR -e '.pgm'

If a file can not be found, its filename will be printed in the standard 
output stream.

.. Place here your external references

.. _mobio: http://www.idiap.ch/dataset/mobio
.. _sqlite: http://www.sqlite.org/
