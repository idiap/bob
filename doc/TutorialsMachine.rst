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

This machine executes the simple operation :math:`y = \mathbf{W} x`, where `y` is
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
  >>> x = numpy.array([0.3, 0.4], 'float64')
  >>> y = machine(x)
  >>> y
  array([ 0.55,  0.55])

As it was shown, the way to pass data through a machine is to call its ``()``
operator.

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

.. note::

  In the event you save a machine that has the subtraction and/or a division
  factor set, the vectors are saved and restored automatically w/o user
  intervention.

The shape of a ``LinearMachine`` indicates the size of the input vector that is
expected by this machine and the size of the output vector it produces, in a
tuple formatted like ``(input-size, output-size)``:

.. doctest::

  >>> machine.shape
  (2, 2)

The ``LinearMachine`` also supports pre-setting normalization vectors that are
applied to every input `x`. You can set a subtraction factor and a division
factor, so that the actual input `x'` that is fed to the matrix `W` is 
:math:`x' = (x .- S) ./ D`. `S` and `D` are vectors that have to have the same
size as the input vector `x`. The operations `.-` and `./` indicate
element-wise subtraction and division respectively. By default, 
:math:`S := 0.0` and :math:`D := 1.0`.

.. doctest::

  >>> machine.input_subtract
  array([ 0.,  0.])
  >>> machine.input_divide
  array([ 1.,  1.])

To set a new value, just assign to the machine property:

.. doctest::

  >>> machine.input_subtract = numpy.array([0.5, 0.8])
  >>> machine.input_divide = numpy.array([2.0, 4.0])
  >>> y = machine(x)
  >>> y
  array([-0.15, -0.15])

You will find interesting ways to train ``LinearMachines`` so they can do
something useful to you at :doc:`TutorialsTrainer`.

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
