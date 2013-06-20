#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 13 Jun 2013 15:25:15 CEST 

"""Pythonic implementations of Multi-Layer Perceptrons for code testing
"""

import numpy
from ...machine.test.mlp import Machine as BaseMachine

class Machine(BaseMachine):
  """Represents a Multi-Layer Perceptron Machine with a single hidden layer"""

  def backward(self, b):
    """Executes the backward step for training.

    In this phase, we calculate the error on the output layer and then use
    back-propagation to estimate the error on the hidden layer. We then use
    this estimated error to calculate the differences between what the layer
    output and the expected value.

    Keyword attributes:

    b
      This is the error back-propagated through the last neuron by any of the
      available :py:class:`bob.trainer.Cost` functors. Every row in b matches
      one example.

    self.d

      The updates for each synapse are simply the multiplication of the a's and
      errors's on each end. One important remark to get this computation right:
      one must generate a weight change matrix that is of the same size as the
      weight matrix. If that is not the case, something is wrong on the logic

      self.d[L] = self.a[L-1] * self.b[L].T / number-of-examples

      N.B.: This **is** a matrix multiplication, despite the fact that ``a``
      and ``b`` are vectors.

    Returns the derivative estimations for every weight in the network
    """

    self.b = [b]

    # calculate the estimated errors on each layer ($\delta$)
    for k,w in reversed(list(enumerate(self.weights[1:]))):
      if self.has_bias:
        delta = numpy.dot(self.b[0], w[1:].T)
        act = self.a[k+1][:,1:]
      else:
        delta = numpy.dot(self.b[0], w.T)
        act = self.a[k+1]
      self.b.insert(0, delta*self.hidden_activation.f_prime_from_f(act))

    self.d = []
    for a,b in zip(self.a[:-1], self.b):
      self.d.append(numpy.dot(a.T, b) / len(b))

    return self.d
  
  def cost(self, cost_object, target):
    """Calculates the cost given the target.

    This method must be called after `forward` has been called.
    """
  
    return cost_object.f(self.a[-1], target).mean(axis=0).sum()

  def unroll(self):
    """Unroll its own parameters so it becomes a linear vector"""

    return numpy.hstack([k.flat for k in self.weights])

  def roll(self, v):
    """Roll-up the parameters again, undoes ``unroll()`` above."""
  
    retval = []
    marks = list(numpy.cumsum([k.size for k in self.weights]))
    marks.insert(0, 0)
    
    for k,w in enumerate(self.weights):
      retval.append(v[marks[k]:marks[k+1]].reshape(w.shape))

    return retval
