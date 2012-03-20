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
multi-layer perceptrons or Gaussian-mixtures. The operation you normally expect
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

Linear Machine
--------------

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

Neural Networks: Multi-layer Perceptrons (MLP)
----------------------------------------------

A `multi-layer perceptron <http://en.wikipedia.org/wiki/Multilayer_perceptron>`_
is a neural network architecture that has some well-defined characteristics
such as a feed-forward structure. You can create a new MLP using one of the
trainers described at :doc:`TutorialsTrainer`. In this tutorial, we show only
how to use an MLP.  To instantiate a new (uninitialized)
:py:class:`bob.machine.MLP` pass a shape descriptor as a :py:func:`tuple`. The
shape parameter should contain the input size as the first parameter and the
output size as the last parameter.  The parameters in between define the number
of neurons in the hidden layers of the MLP. For example ``(3, 3, 1)`` defines
an MLP with 3 inputs, 1 single hidden layer with 3 neurons and 1 output,
whereas a shape like ``(10, 5, 3, 2)`` defines an MLP with 10 inputs, 5 neurons
in the first hidden layer, 3 neurons in the second hidden layer and 2 outputs.
Here is an example:

.. doctest::

  >>> mlp = bob.machine.MLP((3, 3, 2, 1))

As it is, the network is uninitialized. For the sake of examplifying how to use
MLPs, let's set the weight and biases manually (we would normally use a trainer
for that):

.. doctest::

  >>> input_to_hidden0 = numpy.ones((3,3), 'float64')
  >>> input_to_hidden0
  array([[ 1.,  1.,  1.],
         [ 1.,  1.,  1.],
         [ 1.,  1.,  1.]])
  >>> hidden0_to_hidden1 = 0.5*numpy.ones((3,2), 'float64')
  >>> hidden0_to_hidden1
  array([[ 0.5,  0.5],
         [ 0.5,  0.5],
         [ 0.5,  0.5]])
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
  >>> mlp.weights = (input_to_hidden0, hidden0_to_hidden1, hidden1_to_output)
  >>> mlp.biases = (bias_hidden0, bias_hidden1, bias_output)

A few notes are due at this point:

1. Weights should **always** be 2D arrays, even if they are connecting 1 neuron
   to many (or many to 1). You can use the NumPy_ ``reshape()`` array method
   for this purpose as shown above
2. Biases should **always** be 1D arrays.
3. By default, MLPs use the `hyperbolic tangent <http://mathworld.wolfram.com/HyperbolicTangent.html>`_ as activation functions.
   Other 2 activation functions are possible:

   * The identity function: :py:const:`bob.machine.Activation.LINEAR`
   * The sigmoid or `logistic function <http://mathworld.wolfram.com/SigmoidFunction.html>`_: :py:const:`bob.machine.Activation.SIGMOID` or 
     :py:const:`bob.machine.Activation.LOG`.

Let's try changing all activation functions for a simpler one, just for this
example:

.. doctest::

  >>> mlp.activation = bob.machine.Activation.LINEAR

Once the network weights and biases are set, we can feed forward an example
through this machine. This is done using the ``()`` operator, like for
a :py:class:`bob.machine.LinearMachine`:

.. doctest::

  >>> mlp(numpy.array([0.1, -0.1, 0.2], 'float64'))
  array([ 0.33])

Support Vector Machines
-----------------------

The :py:class:`bob.machine.SupportVector` implements a Support Vector Machine
with a bridge to `LIBSVM`_. The bridge functionality includes loading and
saving SVM data files and machine models, which you can produce or download
following the instructions found on `LIBSVM`_'s home page. |project| bindings
to `LIBSVM`_ do not allow you to explicitly set the machine's internal values.
You must use the associated trainer as explained on :doc:`TutorialsTrainer` to
generate a valid :py:class:`bob.machine.SupportVector`. Once you have followed
the instructions at :doc:`TutorialsTrainer`, you can come back to this page and
follow the remaining instructions here.

.. note:: 

  Our current ``svm`` object was trained with the file called `heart_scale`,
  distributed with `LIBSVM`_ and `available here
  <http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart_scale>`_.
  This dataset proposes a binary classification problem (i.e., 2 classes of
  features to be discriminated). The number of features is 13.

Our extensions to `LIBSVM`_ also allow you to feed data through a
:py:class:`bob.machine.SupportVector` using :py:class:`numpy.ndarray` objects
and collect results in that format. For the following lines, we assume you have
available a :py:class:`bob.machine.SupportVector` named ``svm``. (For this
example, the variable ``svm`` was generated from the ``heart_scale`` dataset
using the application ``svm-train`` with default parameters.)

.. testsetup:: svm

  import os
  import bob
  import numpy
  
  # the CMAKE_SOURCE_DIR is defined at conf.py.in
  heart_model = os.path.join(os.environ['CMAKE_SOURCE_DIR'], 
    'python/machine/data/heart.svmmodel')

  svm = bob.machine.SupportVector(heart_model)

.. doctest:: svm

  >>> svm.shape
  (13, 1)

To run a single example through the SVM, just use the ``()`` operator like
before:

.. doctest:: svm

  >> svm(numpy.ones((13,), 'float64'))
  1
  >> svm(numpy.ones((10,13), 'float64'))
  (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

Visit the documentation for :py:class:`bob.machine.SupportVector` to find more
information about these bindings and methods you can call on such machine.
Visit the documentation for :py:class:`bob.machine.SVMFile` for information on
loading `LIBSVM`_ data files direction into python and producing
:py:class:`numpy.ndarray` objects.

Here is quick usage example: Suppose the variable ``f`` contains an object of
type :py:class:`bob.machine.SVMFile`. Then, you could read data (and labels)
from the file like this:

.. testsetup:: svmfile

  import os
  import numpy
  import bob

  # the CMAKE_SOURCE_DIR is defined at conf.py.in
  heart_data = os.path.join(os.environ['CMAKE_SOURCE_DIR'], 
    'python/machine/data/heart.svmdata')

  f = bob.machine.SVMFile(heart_data)

  # the CMAKE_SOURCE_DIR is defined at conf.py.in
  heart_model = os.path.join(os.environ['CMAKE_SOURCE_DIR'], 
    'python/machine/data/heart.svmmodel')

  svm = bob.machine.SupportVector(heart_model)

.. doctest:: svmfile

  >>> labels, data = f.read_all()
  >>> data = numpy.vstack(data) #creates a single 2D array

Then you can throw the data into the ``svm`` machine you trained earlier like
this:

.. doctest:: svmfile

  >>> predicted_labels = svm(data) 

As a final note, if you decide to use our `LIBSVM`_ bindings for your
publication, be sure to also cite:

.. code-block:: latex

  @article{CC01a,
   author  = {Chang, Chih-Chung and Lin, Chih-Jen},
   title   = {{LIBSVM}: A library for support vector machines},
   journal = {ACM Transactions on Intelligent Systems and Technology},
   volume  = {2},
   issue   = {3},
   year    = {2011},
   pages   = {27:1--27:27},
   note    = {Software available at \url{http://www.csie.ntu.edu.tw/~cjlin/libsvm}}
  }

Gaussian Machines
-----------------

The :py:class:`bob.machine.Gaussian` represents a `multivariate diagonal
Gaussian (or normal) distribution
<http://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_. The
*diagonality* of the Gaussians in this multivariate distribution refers to the
covariance matrix of the distribution. When the covariance matrix is diagonal,
each variable in the distribution is independent of the others. 

Objects of this class are normally used as building blocks of more complex
:py:class:`bob.machine.GMMMachine` (Gaussian Mixture Model) objects, but can
also be used individually. Here is how to create one multivariate diagonal
Gaussian distribution:

.. doctest::

  >>> g = bob.machine.Gaussian(2) #bi-variate diagonal normal distribution
  >>> g.mean = numpy.array([0.3, 0.7], 'float64')
  >>> g.mean
  array([ 0.3,  0.7])
  >>> g.variance = numpy.array([0.2, 0.1], 'float64')
  >>> g.variance
  array([ 0.2,  0.1])

Once the :py:class:`bob.machine.Gaussian` has been set, you can use it to
estimate the logarithm likelihood of an input feature vector with a matching
number of dimensions:

.. doctest::

  >>> log_likelihood = g(numpy.array([0.4, 0.4], 'float64'))

As with other machines you can save and re-load machines of this type using
:py:meth:`bob.machine.Gaussian.save` and the class constructor respectively.

Gaussian Mixture Models
-----------------------

The :py:class:`bob.machine.GMMMachine` represents a Gaussian 
`Mixture Model <http://en.wikipedia.org/wiki/Mixture_model>`_ (GMM), which
consists in a mixture of weighted :py:class:`bob.machine.Gaussian`.

.. doctest::

  >>> gmm = bob.machine.GMMMachine(2,3) # Mixture of two diagonal Gaussian of dimension 3

By default, the diagonal Gaussian distributions of the GMM are initialized 
with zero mean and unit variance, and the weights are identical. This can be
updated using the :py:attr:`bob.machine.GMMMachine.means`, :py:attr:`bob.machine.GMMMachine.variances`
or :py:attr:`bob.machine.GMMMachine.weights`.

.. doctest::
  :options: +NORMALIZE_WHITESPACE

  >>> gmm.means = numpy.array([[5., 5., 5.], [5., 5., -5.]], 'float64')
  >>> gmm.means
  array([[ 5., 5., 5.], 
         [ 5., 5., -5.]])

Once the :py:class:`bob.machine.GMMMachine` has been set, you can use it to
estimate the logarithm likelihood of an input feature vector with a matching
number of dimensions:

.. doctest::

  >>> log_likelihood = gmm(numpy.array([5.1, 4.7, -4.9], 'float64'))

As with other machines you can save and re-load machines of this type using
:py:meth:`bob.machine.GMMMachine.save` and the class constructor respectively.


.. testcleanup:: *

  import shutil
  os.chdir(current_directory)
  shutil.rmtree(temp_dir)

.. Place here your external references

.. _numpy: http://numpy.scipy.org
.. _libsvm: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
