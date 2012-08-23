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

==========
 Database
==========

|project| provides an API to easily query and interface with well known
databases. A |project| database contains information about the organization
of the files, functions to query information such as the data which might be
used for training a model, but it usually does **not** contain the data itself
(except for some toy examples). Most of the databases are stored in a sqlite_ 
file, whereas the smallest ones are stored as filelists.

As databases usually contain thousands of files, and as verification protocols 
often require to store information about pairs of files, the size of such
databases can become very large. For this reason, we have decided to externalize
many of them in some `Satellite Packages`_.


.. testsetup:: *

   import bob


Iris Flower Data Set
====================

The `Iris flower data set <http://en.wikipedia.org/wiki/Iris_flower_data_set>`_
or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald
Aylmer Fisher (1936) as an example of discriminant analysis. The dataset 
consists of 50 samples from three species of Iris flowers (Iris setosa, Iris 
virginica and Iris versicolor). Four features were measured from each sample, 
they are the length and the width of sepal and petal, in centimeters.

As this data set is quite small and used for testing purpose, it is directly
integrated into |project|, which provides both ways to access the data, as well
as the data itself (feature vectors of length four for various samples of the 
three species).

A description of the feature vector can be obtained using the attribute
:py:attr:`bob.db.iris.names`.

.. doctest::

   >>> descriptor_labels = bob.db.iris.names
   >>> descriptor_labels
   ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

The data (feature vectors) can be retrieved using the :py:meth:`bob.db.iris.data()` 
function. This returns a 3-key dictionary, with 3 :pyth:attr:`bob.io.Arrayset` 
as values, one for each of the three species of Iris flowers.

.. doctest::

   >>> bob.db.iris.data()
   {'setosa': <Arrayset[50] float64@(4,)>, 'versicolor': <Arrayset[50] float64@(4,)>, 'virginica': <Arrayset[50] float64@(4,)>}

Each :pyth:attr:`bob.io.Arrayset` consists of 50 feature vectors of length four.

The database also contains statistics about the feature vectors, which can be 
obtained using the :py:attr:`bob.db.iris.stats` dictionary. A description
of these statistics is provided by :py:attr:`bob.db.iris.stat_names`.

.. include:: links.rst

.. Place here your external references

.. _sqlite: http://www.sqlite.org/
