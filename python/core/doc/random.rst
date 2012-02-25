.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Laurent El Shafey <laurent.el-shafey@idiap.ch>
.. Fri Aug 12 13:36:45 2011 +0200
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

==========================
 Random Number Generation
==========================

Introduction
------------

We have developed a set of bridges to the `Boost Random Number Generation`_
facilities. This allows you to generate random numbers in a variety of ways.

Example:

.. code-block:: python

  >>> mt = bob.core.random.mt19937()
  >>> binom = bob.core.random.binomial_float64()
  >>> binom(mt)
      0 

Please note that Numpy_ also provides random sampling functionalities.

Reference Manual
----------------

.. automodule:: bob.core.random
  :members:

.. Place here your references
.. _numpy: http://numpy.scipy.org
.. _boost random number generation: http://www.boost.org/doc/libs/release/libs/random/index.html 
