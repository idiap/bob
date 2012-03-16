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
 Trainers
**********

Introduction
============

In the previous section, the concept of `machine` has been introduced. A
`machine` is fed by some input data, processes it and returns an output.
For instance, if we consider a **LinearMachine**, the input data :math:`x` 
is projected using an internal projection matrix :math:`\mathbf{W}` which is a 
member of the **LinearMachine** class, and the projected data 
:math:`y = \mathbf{W} x` are returned. Very often, we would like to `learn`
the parameters of a `machine` from the data. This is the role of what 
is referred to as a `trainer` in |project|. Some `machines` might be 
trained using different techniques. For instance, the projection matrix
:math:`\mathbf{W}` of a **LinearMachine** could be learned using 
Principal Component Analysis (PCA) or Linear Discriminant Analysis (LDA).

.. testsetup:: *

   import bob
   import numpy


Principal Component Analysis
============================

`PCA` is one way to train a **LinearMachine**. The associated |project| class
is **SVDPCATrainer** as the training procedure mainly relies on a singular
value decomposition.

The procedure to train a **LinearMachine** with a **SVDPCATrainer** is shown 
below. Please note that the concepts remains really similar for most of the 
other `trainer`/`machines`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> trainer = bob.trainer.SVDPCATrainer()
   >>> data = bob.io.Arrayset()  # Creates a container for the training data
   >>> a = numpy.array([3,-3,100], 'float64')
   >>> b = numpy.array([4,-4,50], 'float64')
   >>> c = numpy.array([3.5,-3.5,-50], 'float64')
   >>> d = numpy.array([3.8,-3.7,-100], 'float64')
   >>> data.append(a)
   >>> data.append(b)
   >>> data.append(c)
   >>> data.append(d)
   >>> print data
   <Arrayset[4] float64@(3,)>
   >>> [machine, eig_vals] = trainer.train(data)  # Trains the machine with the given data
   >>> print machine.weights  # The new weights after the training procedure
   [[  2.20006252e-03 -7.06111790e-01 -7.08096957e-01]
    [ -1.80006431e-03  7.08094727e-01 -7.06115159e-01]
    [ -9.99995960e-01 -2.82811755e-03 -2.86806039e-04]]

Next, input data can be projected using the learned projection matrix 
:math:`W`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> e = numpy.array([3.2,-3.3,-10], 'float64')
   >>> print machine(e)
   [ 9.99868456 0.47009678 0.09187563]


Linear Discrimant Analysis
==========================

`LDA` is another way to train a **LinearMachine**. The associated |project| 
class is **FisherLDATrainer**.

The procedure to train a **LinearMachine** with a **FisherLDATrainer** is shown 
below.

.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
   >>> trainer = bob.trainer.FisherLDATrainer()
   >>> data1 = bob.io.Arrayset()  # Creates a container for the training data of class 1
   >>> a1 = numpy.array([3,-3,100], 'float64')
   >>> b1 = numpy.array([4,-4,50], 'float64')
   >>> c1 = numpy.array([40,-40,150], 'float64')
   >>> data1.append(a1)
   >>> data1.append(b1)
   >>> data1.append(c1)
   >>> data2 = bob.io.Arrayset()  # Creates a container for the training data of class 2
   >>> a2 = numpy.array([3,6,-50], 'float64')
   >>> b2 = numpy.array([4,8,-100], 'float64')
   >>> c2 = numpy.array([40,79,-800], 'float64')
   >>> data2.append(a1)
   >>> data2.append(b2)
   >>> data2.append(c2)
   >>> data = [data1,data2]
   >>> print data
   [<Arrayset[3] float64@(3,)>, <Arrayset[3] float64@(3,)>]
   >>> [machine,eig_vals] = trainer.train(data)  # Trains the machine with the given data
   >>> print eig_vals
   [ 1.93632491 0. ]
   >>> print machine.weights  # The new weights after the training procedure
   [[ 0.83885757 1. ]
    [ 0.53244291 0. ]
    [ 0.11323656 0. ]]
   >>> machine.resize(3,1)  # Make the output space of dimension 1
   >>> print machine.weights  # The new weights after the training procedure
   [[ 0.83885757]
    [ 0.53244291]
    [ 0.11323656]]


Expectation-Maximization for k-Means
====================================


Expectation-Maximization for Gaussian Mixture Model
===================================================


MAP-adaptation for Gaussian Mixture Model
===================================================


Joint Factor Analysis Trainer
=============================


Multi-Layer Perceptron Trainer
==============================


Support Vector Machine Trainer
==============================

.. Place here your external references
