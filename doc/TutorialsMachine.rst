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
consult :doc:`TutorialsIO`). To load a machine from an HDF5 file, you have to
open it first:

MLP
---

SVM
---

GaussianMachine
---------------

GMMMachine
----------

.. Place here your external references

.. _numpy: http://numpy.scipy.org
