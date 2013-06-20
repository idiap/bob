.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Fri 23 Mar 2012 11:41:00 CET
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

=============================================================
 A Complete Application: Analysis of the Fisher Iris Dataset
=============================================================

The `Iris flower data set <http://en.wikipedia.org/wiki/Iris_flower_data_set>`_
or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald
Aylmer Fisher (1936) as an example of discriminant analysis. It is sometimes
called Anderson's Iris data set because Edgar Anderson collected the data to
quantify the morphologic variation of Iris flowers of three related species.
The dataset consists of 50 samples from each of three species of Iris flowers
(Iris setosa, Iris virginica and Iris versicolor). Four features were measured
from each sample, they are the length and the width of sepal and petal, in
centimeters. Based on the combination of the four features, Fisher developed a
linear discriminant model to distinguish the species from each other.

In this example, we collect bits and pieces of the previous tutorials and build
a complete example that discriminates Iris species based on |project|.

.. note::

  This example will consider all 3 classes for the LDA training. This is *not*
  what Fisher did in his paper entitled "The Use of Multiple Measurements in
  Taxonomic Problems", Annals of Eugenics, pp. 179-188, 1936. In that work
  Fisher did the "right" thing only for the first 2-class problem (setosa
  *versus* versicolor). You can reproduce the 2-class LDA using bob's LDA
  training system without problems. When inserting the virginica class, Fisher
  decides for a different metric (:math:`4vi + ve - 5se`) and solves for the
  matrices in the last row of Table VIII.

  This is OK, but does not generalize the method proposed in the beginning of
  his paper. Results achieved by the generalized LDA method [1]_ will not match
  Fisher's result on that last table, be aware. That being said, the final
  histogram presented on that paper looks quite similar to the one produced by
  this script, showing that Fisher's solution was a good approximation for the
  generalized LDA implementation available in |project|.

The Iris dataset
----------------

.. testsetup:: iris

  import bob
  import numpy
  import matplotlib
  if not hasattr(matplotlib, 'backends'):
    matplotlib.use('pdf') #non-interactive avoids exception on display

The Iris dataset is built into |project|. Currently, it is the only dataset
completely available with the source code (you will need to download the other
ones yourself as explained at :doc:`TutorialsDatabase`). The reference manual
for this dataset is available at :py:mod:`bob.db.iris`. The most important
method is :py:func:`bob.db.iris.data`, that returns a standard python
dictionary containing one 2D :py:func:`numpy.ndarray` for each class.  Each
:py:func:`numpy.ndarray` contains the 4 features described in the database for
the given Iris type.

.. doctest:: iris

  >>> data = bob.db.iris.data()
  >>> type(data['setosa'])
  <type 'numpy.ndarray'>
  >>> data['setosa'].shape
  (50, 4)
  >>> data.keys()
  ['setosa', 'versicolor', 'virginica']

  >>> #the features in each array, if you are curious!
  >>> bob.db.iris.names
  ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

Training a :py:class:`bob.machine.LinearMachine` with LDA
---------------------------------------------------------

Creating a :py:class:`bob.machine.LinearMachine` to perform Linear Discriminant
Analysis on the Iris dataset involves using the
:py:class:`bob.trainer.FisherLDATrainer`, as introduced at
:doc:`TutorialsTrainer`.

.. doctest:: iris

  >>> trainer = bob.trainer.FisherLDATrainer()
  >>> machine, unused_eigen_values = trainer.train(data.values())
  >>> machine
  <LinearMachine float64@(4, 2)>

That is it! The returned :py:class:`bob.machine.LinearMachine` is now setup to
perform LDA on the Iris dataset. A few things should be noted:

1. The returned :py:class:`bob.machine.LinearMachine` represents the linear
   projection of the input features to a new 3D space which maximizes the
   between-class scatter and minimizes the within-class scatter. In other
   words, the internal matrix :math:`\mathbf{W}` is 4-by-2. The projections are
   calculated internally using `Singular Value Decomposition
   <http://en.wikipedia.org/wiki/Singular_value_decomposition>`_ (SVD). The
   first projection (first row of :math:`\mathbf{W}` corresponds to the highest
   Eigen value resulting from the decomposition, the second, the second
   highest, and so on;
2. The trainer also returns the eigen values generated after the SVD
   for our LDA implementation, in case you would like to use them. For this
   example, we just discard this information.

Looking at the first LDA component
----------------------------------

To reproduce Fisher's results, we must pass the data through the created
machine:

.. doctest:: iris

  >>> output = {}
  >>> for key in data:
  ...   output[key] = machine.forward(data[key])
  ...

At this point the variable ``output`` contains the LDA-projected information as
2D :py:class:`numpy.ndarray` objects.  The only step missing is the
visualization of the results. Fisher proposed the use of a histogram showing
the separation achieved by looking at the first only.  Let's reproduce it.

.. doctest:: iris

  >>> from matplotlib import pyplot
  >>> pyplot.hist(output['setosa'][:,0], bins=8, color='green', label='Setosa', alpha=0.5) # doctest: +SKIP
  >>> pyplot.hist(output['versicolor'][:,0], bins=8, color='blue', label='Versicolor', alpha=0.5) # doctest: +SKIP
  >>> pyplot.hist(output['virginica'][:,0], bins=8, color='red', label='Virginica', alpha=0.5) # doctest: +SKIP

We can certainly throw in more decoration:

.. doctest:: iris

  >>> pyplot.legend() # doctest: +SKIP
  >>> pyplot.grid(True) # doctest: +SKIP
  >>> pyplot.axis([-3,+3,0,20]) # doctest: +SKIP
  >>> pyplot.title("Iris Plants / 1st. LDA component") # doctest: +SKIP
  >>> pyplot.xlabel("LDA[0]") # doctest: +SKIP
  >>> pyplot.ylabel("Count") # doctest: +SKIP

Finally, to display the plot, do:

.. code-block:: python

  >>> pyplot.show()

You should see an image like this:

.. plot:: plot/iris_lda.py

Measuring performance
---------------------

You can measure the performance of the system on classifying, say, *Iris
Virginica* as compared to the other two variants. We can use the functions in
:py:mod:`bob.measure` for that purpose. Let's first find a threshold that
separates this variant from the others. We choose to find the threshold at the
point where the relative error rate considering both *Versicolor* and *Setosa*
variants is the same as for the *Virginica* one.

.. doctest:: iris

  >>> negatives = numpy.vstack([output['setosa'], output['versicolor']])[:,0]
  >>> positives = output['virginica'][:,0]
  >>> t = bob.measure.eer_threshold(negatives, positives)

With the threshold at hand, we can estimate the number of correctly classified
*negatives* (or true-rejections) and *positives* (or true-accepts). Let's
translate that: plants from the *Versicolor* and *Setosa* variants that have
the first LDA component smaller than the threshold (so called *negatives* at
this point) and plants from the *Virginica* variant that have the first LDA
component greater than the threshold defined (the *positives*). To calculate
the rates, we just use :py:mod:`bob.measure` again:

.. doctest:: iris

  >>> true_rejects = bob.measure.correctly_classified_negatives(negatives, t)
  >>> true_accepts = bob.measure.correctly_classified_positives(positives, t)

From that you can calculate, for example, the number of misses at the defined
threshold "t":

.. doctest:: iris

  >>> sum(true_rejects)
  98
  >>> sum(true_accepts)
  49

You can also plot an ROC curve as explained at :doc:`TutorialsPerformance`.
Here is the full code that will lead you to the following plot:

.. plot:: plot/iris_lda_roc.py
  :include-source: True

.. include:: links.rst

.. some references

.. [1] Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis. (Q327.D83) John Wiley & Sons. ISBN 0-471-22361-1. See page 218.
