.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
.. Wed Mar 14 12:31:35 2012 +0100
.. 
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

In the previous section, the concept of a `machine` was introduced. A
`machine` is fed by some input data, processes it and returns an output.
For instance, if we consider a :py:class:`bob.machine.LinearMachine`, the 
input data :math:`x` is projected using an internal projection matrix 
:math:`\mathbf{W}`, and the projected data :math:`y = \mathbf{W} x` is
returned. Very often, we would like to `learn` the parameters of a `machine`
from the data. This is the role of what is referred to as a `trainer` in 
|project|. Some `machines` might be trained using different techniques. 
For instance, the projection matrix :math:`\mathbf{W}` of a 
:py:class:`bob.machine.LinearMachine` could be learned using 
Principal component analysis (**PCA** [1]_) or Linear discriminant analysis 
(**LDA** [2]_).

.. testsetup:: *

   import bob
   import numpy
  
   numpy.set_printoptions(precision=3, suppress=True)


Principal component analysis
============================

**PCA** [1]_ is one way to train a :py:class:`bob.machine.LinearMachine`. The
associated |project| class is :py:class:`bob.trainer.SVDPCATrainer` as the 
training procedure mainly relies on a singular value decomposition.

**PCA** belongs to the category of `unsupervised` learning algorithms, which
means that the training data is not labelled. Therefore, the training set can
be represented by a set of features stored in a container. Using |project|, 
this container is a 2D :py:class:`numpy.ndarray`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> data = numpy.array([[3,-3,100], [4,-4,50], [3.5,-3.5,-50], [3.8,-3.7,-100]], dtype='float64')
   >>> print data
   [[   3.    -3.   100. ]
    [   4.    -4.    50. ]
    [   3.5   -3.5  -50. ]
    [   3.8   -3.7 -100. ]]

Once the training set has been defined, the overall procedure to train a 
:py:class:`bob.machine.LinearMachine` with a 
:py:class:`bob.trainer.SVDPCATrainer` is simple and shown below. Please 
note that the concepts remains very similar for most of the other 
`trainers` and `machines`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> trainer = bob.trainer.SVDPCATrainer() # Creates a PCA trainer
   >>> [machine, eig_vals] = trainer.train(data)  # Trains the machine with the given data
   >>> print machine.weights  # The weights of the returned LinearMachine after the training procedure
   [[ 0.002 -0.706 -0.708]
    [-0.002  0.708 -0.706]
    [-1.    -0.003 -0.   ]]

Next, input data can be projected using this learned projection matrix 
:math:`W`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> e = numpy.array([3.2,-3.3,-10], 'float64')
   >>> print machine(e)
   [ 9.999 0.47 0.092]


Linear discriminant analysis
============================

**LDA** [2]_ is another way to train a :py:class:`bob.machine.LinearMachine`. 
The associated |project| class is :py:class:`bob.trainer.FisherLDATrainer`.

In contrast to **PCA** [1]_, **LDA** [2]_ is a `supervised` technique.
Furthermore, the training data should be organized differently. It is indeed 
required to be a list of 2D :py:class:`numpy.ndarray`\'s, one for each class.

.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
   >>> data1 = numpy.array([[3,-3,100], [4,-4,50], [40,-40,150]], dtype='float64')
   >>> data2 = numpy.array([[3,6,-50], [4,8,-100], [40,79,-800]], dtype='float64')
   >>> data = [data1,data2]

Once the training set has been defined, the procedure to train the 
:py:class:`bob.machine.LinearMachine` with **LDA** is very similar to the one
for **PCA**. This is shown below.

.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
   >>> trainer = bob.trainer.FisherLDATrainer()
   >>> [machine,eig_vals] = trainer.train(data)  # Trains the machine with the given data
   >>> print eig_vals  # doctest: +SKIP
   [ 13.10097786 0. ]
   >>> machine.resize(3,1)  # Make the output space of dimension 1
   >>> print machine.weights  # The new weights after the training procedure
   [[ 0.609]
    [ 0.785]
    [ 0.111]]


Neural networks: multi-layer perceptrons (MLP)
==============================================

A **multilayer perceptron** (MLP) [3]_ is a neural network architecture that 
has some well-defined characteristics such as a feed-forward structure. 
As described in :doc:`TutorialsMachine`, an MLP can be created as follows:

.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
   >>> machine = bob.machine.MLP((2, 2, 1)) # Creates a MLP with 2 inputs, 2 neurons in each hidden layer ad 1 output
   >>> machine.activation = bob.machine.Activation.LOG # Uses a log() activation function
   >>> machine.biases = 0 # Set the biases to 0
   >>> w0 = numpy.array([[.23, .1],[-0.79, 0.21]])
   >>> w1 = numpy.array([[-.12], [-0.88]])
   >>> machine.weights = [w0, w1] # Sets the initial weights of the machine

Such a network can be `trained` through backpropagation [4]_, which is 
a supervised learning technique. Therefore, the training procedure requires a
set of features with labels (or targets). Using |project|, this is passed to
the `train()` method in two different 2D `NumPy`_ arrays, one for the input 
(features) and one for the output (targets).

.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
   >>> d0 = numpy.array([[.3, .7]]) # input
   >>> t0 = numpy.array([[.0]]) # target

The class used to train a MLP [3]_ with backpropagation [4]_ is 
:py:class:`bob.trainer.MLPBackPropTrainer`. An example is shown below.


.. doctest::
   :options: +NORMALIZE_WHITESPACE
   
   >>> trainer = bob.trainer.MLPBackPropTrainer(machine, 1) #  Creates a BackProp trainer with a batch size of 1
   >>> trainer.train_biases = False # Do not train the bias
   >>> trainer.train(machine, d0, t0) # Performs the Back Propagation

Backpropagation [4]_ requires a learning rate to be set. In the previous 
example, the default value 0.1 has been used. This might be updated using the
:py:attr:`bob.trainer.MLPBackPropTrainer.learningRate` attribute. Another alternative exists 
referred to as **resilient propagation** (Rprop) [5]_, which dynamically computes an optimal 
learning rate. The corresponding class is :py:class:`bob.trainer.MLPRPropTrainer`, and the 
overall training procedure remains identical.

.. doctest::
   :options: +NORMALIZE_WHITESPACE
 
   >>> trainer = bob.trainer.MLPRPropTrainer(machine, 1)
   >>> trainer.train_biases = False
   >>> trainer.train(machine, d0, t0) 


Support vector machines
=======================

.. ifconfig:: not has_libsvm

  .. warning:: 

    LIBSVM was not found when this documentation has been generated.


A **support vector machine** (SVM) [6]_ is a very popular `supervised` learning 
technique. |project| provides a bridge to `LIBSVM`_ which allows you to `train`
such a `machine` and use it for classification. 

The training set for such a machine consists of a list of 2D `NumPy` arrays,
one for each class. The first dimension of each 2D `NumPy` array is the number
of training samples for the given class and the second dimension is the dimensionality
of the feature. For instance, let's consider the following training set for a two 
class problem.

.. ifconfig:: has_libsvm

  .. doctest::
     :options: +NORMALIZE_WHITESPACE

     >>> pos = numpy.array([[1,-1,1], [0.5,-0.5,0.5], [0.75,-0.75,0.8]], 'float64')
     >>> neg = numpy.array([[-1,1,-0.75], [-0.25,0.5,-0.8]], 'float64')
     >>> data = [pos,neg]
     >>> print data # doctest: +SKIP

.. ifconfig:: not has_libsvm

  .. code-block:: python

     >>> pos = numpy.array([[1,-1,1], [0.5,-0.5,0.5], [0.75,-0.75,0.8]], 'float64')
     >>> neg = numpy.array([[-1,1,-0.75], [-0.25,0.5,-0.8]], 'float64')
     >>> data = [pos,neg]
     >>> print data # doctest: +SKIP

.. note:: 

  Please note that in the above training set, the data is pre-scaled so 
  features remain in the range between -1 and +1. libsvm, apparently, suggests
  you do that for all features. Our bindings to libsvm do not include scaling. 
  If you want to implement that generically, please do it.

Then, an SVM [6]_ can be trained easily using the :py:class:`bob.trainer.SVMTrainer` class.

.. ifconfig:: has_libsvm

  .. doctest::
     :options: +NORMALIZE_WHITESPACE

     >>> trainer = bob.trainer.SVMTrainer()
     >>> machine = trainer.train(data) #ordering only affects labels

.. ifconfig:: not has_libsvm

  .. code-block:: python

     >>> trainer = bob.trainer.SVMTrainer()
     >>> machine = trainer.train(data) #ordering only affects labels

This returns a :py:class:`bob.machine.SupportVector` which can later be used 
for classification, as explained in :doc:`TutorialsMachine`.

.. ifconfig:: has_libsvm

  .. doctest::

     >>> predicted_label = machine(numpy.array([1.,-1.,1.]))
     >>> print predicted_label
     1

.. ifconfig:: not has_libsvm

  .. code-block:: python

     >>> predicted_label = machine(numpy.array([1.,-1.,1.]))
     >>> print predicted_label
     1

The `training` procedure allows several different options. For 
instance, the default `kernel` is an `RBF`. If we would like a `linear SVM` 
instead, this can be set before calling the 
:py:meth:`bob.trainer.SVMTrainer.train()` method.

.. ifconfig:: has_libsvm

  .. doctest::
     :options: +NORMALIZE_WHITESPACE

     >>> trainer.kernel_type = bob.machine.svm_kernel_type.LINEAR

.. ifconfig:: not has_libsvm

  .. code-block:: python

     >>> trainer.kernel_type = bob.machine.svm_kernel_type.LINEAR


K-means
=======

**k-means** [7]_ is a clustering method, which aims to partition a 
set of observations into :math:`k` clusters. This is an `unsupervised` 
technique. As for **PCA** [1]_, the training data is passed
in a 2D :py:class:`numpy.ndarray` container.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> data = numpy.array([[3,-3,100], [4,-4,98], [3.5,-3.5,99], [-7,7,-100], [-5,5,-101]], dtype='float64')

The training procedure will learn the `means` for the :py:class:`bob.machine.KMeansMachine`. The 
number :math:`k` of `means` is given when creating the `machine`, as well as the dimensionality of
the features.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> kmeans = bob.machine.KMeansMachine(2, 3) # Create a machine with k=2 clusters with a dimensionality equal to 3

Then training procedure for `k-means` is an **Expectation-Maximization**-based
[8]_ algorithm. There are several options that can be set such as the maximum
number of iterations and the criterion used to determine if the convergence has
occurred. After setting all of these options, the training procedure can then
be called.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> kmeansTrainer = bob.trainer.KMeansTrainer()
   >>> kmeansTrainer.max_iterations = 200
   >>> kmeansTrainer.convergence_threshold = 1e-5

   >>> kmeansTrainer.train(kmeans, data) # Train the KMeansMachine
   >>> print kmeans.means
   [[ -6.   6.  -100.5]
    [  3.5 -3.5   99. ]]  


Maximum likelihood for Gaussian mixture model
=============================================

A Gaussian **mixture model** (GMM) [9]_ is a common probabilistic model. In 
order to train the parameters of such a model it is common to use a 
**maximum-likelihood** (ML) approach [10]_. To do this we use an 
**Expectation-Maximization** (EM) algorithm [8]_. Let's first start by creating 
a :py:class:`bob.machine.GMMMachine`. By default, all of the Gaussian's 
have zero-mean and unit variance, and all the weights are equal. As a starting 
point, we could set the mean to the one obtained with **k-means** [7]_.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> gmm = bob.machine.GMMMachine(2,3) # Create a machine with 2 Gaussian and feature dimensionality 3
   >>> gmm.means = kmeans.means # Set the means to the one obtained with k-means 

The |project| class to learn the parameters of a GMM [9]_ using ML [10]_ is
:py:class:`bob.trainer.ML_GMMTrainer`. It uses an **EM**-based [8]_ algorithm
and requires the user to specify which parameters of the GMM are updated at each iteration 
(means, variances and/or weights). In addition, and as for **k-means** [7]_,
it has parameters such as the maximum number of iterations and the criterion 
used to determine if the parameters have converged.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> trainer = bob.trainer.ML_GMMTrainer(True, True, True) # update means/variances/weights at each iteration
   >>> trainer.convergence_threshold = 1e-5
   >>> trainer.max_iterations = 200
   >>> trainer.train(gmm, data)
   >>> print gmm # doctest: +SKIP


MAP-adaptation for Gaussian mixture model
=========================================

|project| also supports the training of GMMs [9]_ using a **maximum a
posteriori** (MAP) approach [11]_. MAP is closely related to the ML [10]_
technique but it incorporates a prior on the quantity that we want to estimate.
In our case, this prior is a GMM [9]_. Based on this prior model and some
training data, a new model, the MAP estimate, will be `adapted`.

Let's consider that the previously trained GMM [9]_ is our prior model.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> print gmm # doctest: +SKIP

The training data used to compute the MAP estimate [11]_ is again stored in a
2D :py:class:`numpy.ndarray` container.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> dataMAP = numpy.array([[7,-7,102], [6,-6,103], [-3.5,3.5,-97]], dtype='float64')

The |project| class used to perform MAP adaptation training [11]_ is
:py:class:`bob.trainer.MAP_GMMTrainer`. As with the ML estimate [10]_, it uses
an **EM**-based [8]_ algorithm and requires the user to specify which parts of
the GMM are adapted at each iteration (means, variances and/or weights). In
addition, it also has parameters such as the maximum number of iterations and
the criterion used to determine if the parameters have converged, in addition
to this there is also a relevance factor which indicates the importance we give
to the prior.  Once the trainer has been created, a prior GMM [9]_ needs to be
set.

.. doctest::
   :options: +NORMALIZE_WHITESPACE
  
   >>> relevance_factor = 4.
   >>> trainer = bob.trainer.MAP_GMMTrainer(relevance_factor, True, False, False) # mean adaptation only
   >>> trainer.convergence_threshold = 1e-5
   >>> trainer.max_iterations = 200
   >>> trainer.set_prior_gmm(gmm)
   True
   >>> gmmAdapted = bob.machine.GMMMachine(2,3) # Create a new machine for the MAP estimate
   >>> trainer.train(gmmAdapted, dataMAP)
   >>> print gmmAdapted # doctest: +SKIP


.. Place here your external references

.. include:: links.rst
.. [1] http://en.wikipedia.org/wiki/Principal_component_analysis
.. [2] http://en.wikipedia.org/wiki/Linear_discriminant_analysis
.. [3] http://en.wikipedia.org/wiki/Multilayer_perceptron
.. [4] http://en.wikipedia.org/wiki/Backpropagation
.. [5] http://en.wikipedia.org/wiki/Rprop
.. [6] http://en.wikipedia.org/wiki/Support_vector_machine
.. [7] http://en.wikipedia.org/wiki/K-means_clustering
.. [8] http://en.wikipedia.org/wiki/Expectation-maximization_algorithm
.. [9] http://en.wikipedia.org/wiki/Mixture_model
.. [10] http://en.wikipedia.org/wiki/Maximum_likelihood
.. [11] http://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation
