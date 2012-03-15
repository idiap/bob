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
 Machines
==========

Machines are one of the core components of |project|. They represent
statistical models that are `trainable`. Examples of machines are
multi-layer perceptrons or gaussian-mixtures. The operation you normally expect
from a machine is to be able to feed a feature vector and extract the machine
response or output for that input vector. It works, in many ways, similarly to
signal processing blocks. Different types of machines will get you a different
types of output. In this tutorial we examine a few of the machines available in
|project| and how to make use of them. Let's start by the simplest of the
machines: a ``LinearMachine``.

.. testsetup:: *

   import numpy
   import bob
   import tempfile
   import os

   current_directory = os.path.realpath(os.curdir)
   temp_dir = tempfile.mkdtemp(prefix='bob_doctest_')
   os.chdir(temp_dir)

LinearMachine
-------------

This machine executes the simple operation :math:`y = x \cdot W`, where `y` is
the output vector, `x`, the input vector and `W` a matrix (2D array), stored
inside the machine. The input vector `x` should be composed of double-precision
floating-point elements. The output will also be in double-precision. Here is
how to use a `LinearMachine`:

.. doctest::

  >>> W = numpy.array([[0.5, 0.5], [1.0, 1.0]], 'float64')
  >>> W
  array([[ 0.5,  0.5],
         [ 1. ,  1. ]])
  >>> machine = bob.machine.LinearMachine(W)
  >>> machine.shape
  (2, 2)
  >>> x = [0.3, 0.4]
  >>> y = machine(x)
  >>> y
  array([ 0.55,  0.55])

The first thing to notice about machines is that they can be stored and
retrieved in HDF5 files (for more details in manipulating HDF5 files, please
consult :doc:`TutorialsIO`). To save the beforemetioned machine to a file, just
use the machine's ``save()`` command. Because several machines can be stored on
the same HDF5File, we let the user open the file and set it up before the
machine can write on it:

.. doctest::

  >>> myh5_file = bob.io.HDF5File('linear.hdf5')
  >>> #do other operations on myh5_file to set it up, optionally
  >>> machine.save(myh5_file)
  >>> del myh5_file #close

You can load the machine again in a similar way:

.. doctest::

  >>> myh5_file = bob.io.HDF5File('linear.hdf5')
  >>> reloaded = bob.machine.LinearMachine(myh5_file)
  >>> numpy.array_equal(machine.weights, reloaded.weights)
  True

MLP
---

SVM
---

GaussianMachine
---------------

GMMMachine
----------

.. testcleanup:: *

  import shutil
  os.chdir(current_directory)
  shutil.rmtree(temp_dir)

.. Place here your external references

.. _numpy: http://numpy.scipy.org
