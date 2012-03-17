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
machines: a :py:class:`bob.machine.LinearMachine`.

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

This machine executes the simple operation :math:`y = \mathbf{W} x`, where `y`
is the output vector, `x`, the input vector and `W` a matrix (2D array), stored
inside the machine. The input vector `x` should be composed of double-precision
floating-point elements. The output will also be in double-precision. Here is
how to use a :py:class:`bob.machine.LinearMachine`:

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
use the machine's ``save`` command. Because several machines can be stored on
the same :py:class:`bob.io.HDF5File`, we let the user open the file and set it
up before the machine can write on it:

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

The shape of a ``LinearMachine`` (see
:py:attr:`bob.machine.LinearMachine.shape`) indicates the size of the input
vector that is expected by this machine and the size of the output vector it
produces, in a tuple formatted like ``(input_size, output_size)``:

.. doctest::

  >>> machine.shape
  (2, 2)

A :py:class:`bob.machine.LinearMachine`` also supports pre-setting
normalization vectors that are applied to every input `x`. You can set a
subtraction factor and a division factor, so that the actual input `x'` that is
fed to the matrix `W` is :math:`x' = (x .- S) ./ D`. `S` and `D` are vectors
that have to have the same size as the input vector `x`. The operations `.-`
and `./` indicate element-wise subtraction and division respectively. By
default, :math:`S := 0.0` and :math:`D := 1.0`.

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

.. note::

  In the event you save a machine that has the subtraction and/or a division
  factor set, the vectors are saved and restored automatically w/o user
  intervention.

You will find interesting ways to train a :py:class:`bob.machine.LinearMachine`
so they can do something useful to you at :doc:`TutorialsTrainer`.

MLP
---

A multi-layer perceptron is a neural network architecture that has some
well-defined characteristics such as a feed-forward structure [1]_.  You can
create a new MLP using one of the trainers described at
:doc:`TutorialsTrainer`. In this tutorial, we show only how to use an MLP. To
instantiate a new (uninitialized) :py:class:`bob.machine.MLP` pass a shape
descriptor as a :py:class:`tuple`. The shape parameter should contain the input
size as the first parameter and the output size as the last parameter. The
parameters in between define the number of neurons in the hidden layers of the
MLP. For example ``(3, 3, 1)`` defines an MLP with 3 inputs, 1 single hidden
layer with 3 neurons and 1 output, whereas a shape like ``(10, 5, 3, 2)``
defines an MLP with 10 inputs, 5 neurons in the first hidden layer, 3 neurons
in the second hidden layer and 2 outputs. Here is an example:

.. doctest::

  >>> mlp = bob.machine.MLP((3, 3, 2, 1))

The network is uninitialized, for the sake of examplifying how to use MLPs,
let's set the weight and biases manually:

.. doctest::

  >>> input_to_hidden0 = numpy.array([0.5, 0.3, 0.2, -1.0, 0.6, -0.1, 0.9, 0.8, 0.4], 'float64').reshape((3,3))
  >>> input_to_hidden0
  array([[ 0.5,  0.3,  0.2],
         [-1. ,  0.6, -0.1],
         [ 0.9,  0.8,  0.4]])
  >>> hidden0_to_hidden1 = numpy.array([0.4, 0.2, 0.1, 0.5, 0.6, 0.7], 'float64').reshape(3,2)
  >>> hidden0_to_hidden1
  array([[ 0.4,  0.2],
         [ 0.1,  0.5],
         [ 0.6,  0.7]])
  >>> hidden1_to_output = numpy.array([0.3, 0.2], 'float64').reshape(2,1)
  >>> hidden1_to_output
  array([[ 0.3],
         [ 0.2]])
  >>> bias_hidden0 = numpy.array([-0.2, -0.3, -0.1], 'float64')
  >>> bias_hidden0
  array([-0.2, -0.3, -0.1])
  >>> bias_hidden1 = numpy.array([-0.7, 0.2], 'float64')
  >>> bias_hidden1
  array([-0.7,  0.2])
  >>> bias_output = numpy.array([0.5], 'float64')
  >>> bias_output
  array([ 0.5])
  >>> mlp.weights = [input_to_hidden0, hidden0_to_hidden1, hidden1_to_output]
  >>> mlp.biases = [bias_hidden0, bias_hidden1, bias_output]

A few notes are due at this point:

1. Weights should **always** be 2D arrays, even if they are connecting 1 neuron
   to many (or many to 1). You can use the NumPy_ ``reshape()`` array method
   for this purpose as shown above
2. Biases should **always** be 1D arrays.

Once the network weights and biases are set, we can feed forward an example
through this machine. This is done using the ``()`` operator, like for
a :py:class:`bob.machine.LinearMachines`:

.. doctest::

  >>> 

You can lookup the reference manual for ``MLPs`` if you need to set, for
example, the activation function. By default, ``MLPs`` use a hyperbolic-tangent
as activation function.

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
.. [1] http://en.wikipedia.org/wiki/Multilayer_perceptron
