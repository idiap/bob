.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Fri 23 Mar 2012 11:41:00 CET 
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
  what Fisher did on his paper entitled "The Use of Multiple Measurements in
  Taxonomic Problems", Annals of Eugenics, pp. 179-188, 1936. In that work
  Fisher does the "right" thing only for the first 2-class problem (setosa
  *versus* versicolor). You can reproduce the 2-class LDA using bob's LDA
  training system w/o problems. When inserting the virginica class, Fisher
  decides for a different metric (:math:`4vi + ve - 5se`) and solves  for the
  matrices in the last row of Table VIII.

  This is OK, but does not generalize the method proposed on the begining of
  his paper. Results achieved by the generalized LDA method [1]_ will not match
  Fisher's result on that last table, be aware. That being said, the final
  histogram presented on that paper looks quite similar to the one produced by
  this script, showing that Fisher's solution was a good approximation for the
  generalized LDA implementation available at |project|.

The Iris Dataset
----------------

.. testsetup:: iris

  import bob
  import numpy
  import matplotlib
  matplotlib.use('pdf') #non-interactive avoids exception on display

The Iris Dataset is built into the guts of |project|. It is likely the only
dataset completely available with the source code (you will need to download
the other ones yourself as explained at :doc:`TutorialsDatabase`). The
reference manual for this dataset is available at :py:mod:`bob.db.iris`. The
most important method is :py:func:`bob.db.iris.data`, that returns a standard
python dictionary containing one :py:func:`bob.io.Arrayset` for each class.
Each :py:func:`bob.io.Arrayset` contains the 4 features described in the
database for the given Iris type.

.. doctest:: iris

  >>> data = bob.db.iris.data()
  >>> data
  {'setosa': <Arrayset[50] float64@(4,)>, 'versicolor': <Arrayset[50] float64@(4,)>, 'virginica': <Arrayset[50] float64@(4,)>}

  >>> #the properties in each Arrayset, if you are curious!
  >>> bob.db.iris.names
  ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

Training a :py:class:`bob.machine.LinearMachine` with LDA
---------------------------------------------------------

Creating a :py:class:`bob.machine.LinearMachine` to perform Linear Discriminant
Analysis on the Iris Dataset involves using the
:py:class:`bob.trainer.FisherLDATrainer`, as introduced at
:doc:`TutorialsTrainer`.

.. doctest:: iris

  >>> trainer = bob.trainer.FisherLDATrainer()
  >>> machine, eigen_values = trainer.train(data.values())
  >>> machine
  <LinearMachine float64@(4, 3)>

That is it! The returned :py:class:`bob.machine.LinearMachine` is now setup to
perform LDA on the Iris dataset. A few things should be noted:

1. The returned :py:class:`bob.machine.LinearMachine` represents the linear
   projection of the input features to a new 3D space which maximizes the
   between-class scatter and minimizes the within-class scatter. In other
   words, the internal matrix :math:`\mathbf{W}` is 4-by-3. The projections are
   calculated internally using `Singular Value Decomposition
   <http://en.wikipedia.org/wiki/Singular_value_decomposition>`_ (SVD). The
   first projection (first row of :math:`\mathbf{W}` corresponds to the highest
   Eigen value resulting from the decomposition, the second, the second
   highest, and so on;
2. The trainer also returns the eigen values generated after the SVD
   for our LDA implementation, in case you would like to use them. For this
   example, we just discard this information.

Looking at the First LDA Component
----------------------------------

To reproduce Fisher's results, we must pass the data through the created
machine:

.. doctest:: iris

  >>> output = {}
  >>> for key in data.keys():
  ...   output[key] = data[key].foreach(machine.forward)
  ...
  >>> output
  {'setosa': <Arrayset[50] float64@(3,)>, 'versicolor': <Arrayset[50] float64@(3,)>, 'virginica': <Arrayset[50] float64@(3,)>}

At this point the variable ``output`` contains the LDA-projected information as
:py:class:`bob.io.Arrayset` objects. Let's convert them to a 2D
:py:class:`numpy.ndarray` so it is easy to feed information into `Matplotlib`_.

.. doctest:: iris

  >>> for key, value in output.iteritems():
  ...   output[key] = numpy.vstack(value)
  ...

Now, the only step missing is the visualization of the results. Fisher proposed
an histogram showing the separation achieved by looking at the first only.
Let's reproduce it.

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

.. include:: links.rst

.. some references

.. [1] Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis. (Q327.D83) John Wiley & Sons. ISBN 0-471-22361-1. See page 218.
