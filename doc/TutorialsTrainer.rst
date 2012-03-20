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
For instance, if we consider a :py:class:`bob.machine.LinearMachine`, the 
input data :math:`x` is projected using an internal projection matrix 
:math:`\mathbf{W}`, and the projected data :math:`y = \mathbf{W} x` are 
returned. Very often, we would like to `learn` the parameters of a `machine`
from the data. This is the role of what is referred to as a `trainer` in 
|project|. Some `machines` might be trained using different techniques. 
For instance, the projection matrix :math:`\mathbf{W}` of a 
:py:class:`bob.machine.LinearMachine` could be learned using 
Principal Component Analysis (**PCA** [1]_) or Linear Discriminant Analysis 
(**LDA** [2]_).

.. testsetup:: *

   import bob
   import numpy


Principal Component Analysis
============================

**PCA** [1]_ is one way to train a :py:class:`bob.machine.LinearMachine`. The
associated |project| class is :py:class:`bob.trainer.SVDPCATrainer` as the 
training procedure mainly relies on a singular value decomposition.

**PCA** belongs to the category of `unsupervised` learning algorithms, which
means that the training data is not labelled. Therefore, the training set can
be represented by a set of features stored in a container. Using |project|, 
this container is a :py:class:`bob.io.Arrayset`. 

.. doctest::
   :options: +NORMALIZE_WHITESPACE

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

Once the training set has been defined, the overall procedure to train a 
:py:class:`bob.machine.LinearMachine` with a 
:py:class:`bob.trainer.SVDPCATrainer` is simple and shown below. Please 
note that the concepts remains really similar for most of the other 
`trainer`/`machines`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> trainer = bob.trainer.SVDPCATrainer() # Creates a PCA trainer
   >>> [machine, eig_vals] = trainer.train(data)  # Trains the machine with the given data
   >>> print machine.weights  # The weights of the returned LinearMachine after the training procedure
   [[  2.20006252e-03 -7.06111790e-01 -7.08096957e-01]
    [ -1.80006431e-03  7.08094727e-01 -7.06115159e-01]
    [ -9.99995960e-01 -2.82811755e-03 -2.86806039e-04]]

Next, input data can be projected using this learned projection matrix 
:math:`W`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> e = numpy.array([3.2,-3.3,-10], 'float64')
   >>> print machine(e)
   [ 9.99868456 0.47009678 0.09187563]


Linear Discrimant Analysis
==========================

**LDA** [2]_ is another way to train a :py:class:`bob.machine.LinearMachine`. 
The associated |project| class is :py:class:`bob.trainer.FisherLDATrainer`.

In contrast to **PCA** [1]_, **LDA** [2]_ is a `supervised` technique.
Furthermore, the training data should be organized differently. It is indeed 
required to be a list of :py:class:`bob.io.Arrayset`, one for each class.

.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
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

Once the training set has been defined, the procedure to train the 
:py:class:`bob.machine.LinearMachine` with **LDA** is very similar to the one
for **PCA**. This is shown below.

.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
   >>> trainer = bob.trainer.FisherLDATrainer()
   >>> [machine,eig_vals] = trainer.train(data)  # Trains the machine with the given data
   >>> print eig_vals  # doctest: +SKIP
   [ 1.93632491 0. ]
   >>> machine.resize(3,1)  # Make the output space of dimension 1
   >>> print machine.weights  # The new weights after the training procedure
   [[ 0.83885757]
    [ 0.53244291]
    [ 0.11323656]]


Neural Networks: Multi-layer Perceptrons (MLP)
==============================================

Support Vector Machines
=======================

k-Means
=======

**k-Means** [3]_ is a clustering method, which aims to partition a 
set of observations into :math:`k` clusters. This is an `unsupervised` 
technique. Furthermore, and as for **PCA** [1]_, the training data is passed
in a :py:class:`bob.io.Arrayset` container.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> data = bob.io.Arrayset()  # Creates a container for the training data
   >>> a = numpy.array([3,-3,100], 'float64')
   >>> b = numpy.array([4,-4,98], 'float64')
   >>> c = numpy.array([3.5,-3.5,99], 'float64')
   >>> d = numpy.array([-7,7,-100], 'float64')
   >>> e = numpy.array([-5,5,-101], 'float64')
   >>> data.append(a)
   >>> data.append(b)
   >>> data.append(c)
   >>> data.append(d)
   >>> data.append(e)
   >>> print data
   <Arrayset[5] float64@(3,)>

The training procedure is going to learn the `means` of a 
:py:class:`bob.machine.KMeansMachine`. The number :math:`k` of `means` is
directly given when creating the `machine`, as well as the feature 
dimensionality.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> kmeans = bob.machine.KMeansMachine(2, 3) # Create a machine with k=2 clusters with a dimensionality equal to 3

Then, the parameters of the **Expectation-Maximization**-based [4]_ `trainer`
is set such as the maximum number of iterations and the criterium used to 
determine if the convergence has occurred. Next, the training procedure can be
called.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> kmeansTrainer = bob.trainer.KMeansTrainer()
   >>> kmeansTrainer.maxIterations = 200
   >>> kmeansTrainer.convergenceThreshold = 1e-5

   >>> kmeansTrainer.train(kmeans, data) # Train the KMeansMachine
   >>> print kmeans.means
   [[ -6.   6.  -100.5]
    [  3.5 -3.5   99. ]]  


Maximum Likelihood for Gaussian Mixture Model
=============================================

Gaussian **Mixture Model** (GMM) [5]_ is a common probabilistic model. In this
context, there is often a need to tune the parameters of such a model given 
some training data. For this purpose, the **maximum-likelihood** technique 
(ML) [6]_ can be applied.
Let's first start by creating a :py:class:`bob.machine.GMMMachine`. By default,
its Gaussian have zero-mean and unit variance, and all the weights are equal.
As a starting point, we could set the mean to the one obtained with 
**k-means** [3]_.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> gmm = bob.machine.GMMMachine(2,3) # Create a machine with 2 Gaussian and feature dimensionality 3
   >>> gmm.means = kmeans.means # Set the means to the one obtained with k-means 

The |project| class to perform **maximum-likelihood** [6]_ for a GMM [5]_ is
:py:class:`bob.trainer.ML_GMMTrainer`. It uses an **EM**-based [4]_ algorithm
and requires to specify which parts of the GMM are updated at each iteration 
(means, variances and/or weights). In addition, and as for **k-means** [3]_,
it has parameters such as the maximum number of iterations and the criterium 
used to determine if the convergence has occurred.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> trainer = bob.trainer.ML_GMMTrainer(True, True, True) # update means/variances/weights at each iteration
   >>> trainer.convergenceThreshold = 1e-5
   >>> trainer.maxIterations = 200
   >>> trainer.train(gmm, data)
   >>> print gmm # doctest: +SKIP
   Weights = (0,1)
  [ 0.4 0.6 ]
  <BLANKLINE>
  Gaussian 0: 
  Mean = (0,2)
  [ -6 6 -100.5 ]
  <BLANKLINE>
  Variance = (0,2)
  [ 1 1 0.25 ]
  <BLANKLINE>
  Gaussian 1: 
  Mean = (0,2)
  [ 3.5 -3.5 99 ]
  <BLANKLINE>
  Variance = (0,2)
  [ 0.166667 0.166667 0.666667 ]


MAP-adaptation for Gaussian Mixture Model
=========================================





Joint Factor Analysis Trainer
=============================


.. Place here your external references

.. [1] http://en.wikipedia.org/wiki/Principal_component_analysis
.. [2] http://en.wikipedia.org/wiki/Linear_discriminant_analysis
.. [3] http://en.wikipedia.org/wiki/K-means_clustering
.. [4] http://en.wikipedia.org/wiki/Expectation-maximization_algorithm
.. [5] http://en.wikipedia.org/wiki/Mixture_model
.. [6] http://en.wikipedia.org/wiki/Maximum_likelihood
