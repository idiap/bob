#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 13 Jun 2013 15:54:19 CEST 

"""Pythonic implementations of Multi-Layer Perceptrons for code testing
"""

import numpy

class Machine:
  """Represents a Multi-Layer Perceptron Machine with a single hidden layer"""

  def __init__(self, bias, weights, hidden_activation, output_activation):
    """Initializes the MLP with a number of inputs and outputs. Weights are
    initialized randomly with the specified seed.
    
    Keyword parameters:

    bias
      A list of 1D numpy.ndarray's with 64-bit floating-point numbers
      representing the biases for each layer of the MLP. Each ndarray must have
      as many entries as neurons in that particular layer. If set to `None`,
      disables the use of biases.

    weights
      A list of 2D numpy.ndarray's with 64-bit floating-point numbers
      representing the weights for the MLP. The more entries, the more layers
      the MLP has. The weight matrix includes the bias terms weights and is
      organized so that every neuron input is in a single column. The first
      row always represents the bias connections.

    hidden_activation
      The activation function to use for the hidden neurons of the network.
      Should be one of the classes derived from
      :py:class:`bob.machine.Activation`.

    output_activation
      The activation function to use for the output neurons of the network.
      Should be one of the classes derived from
      :py:class:`bob.machine.Activation`.
    """

    if bias is None:
      self.weights = weights
      self.has_bias = False
    else:
      self.weights = [numpy.vstack([bias[k], weights[k]]) for k in range(len(bias))]
      self.has_bias = True

    self.hidden_activation = hidden_activation
    self.output_activation = output_activation

  def forward(self, X):
    """Executes the forward step of the N-layer neural network.

    Remember that:

    1. z = X . w

    and

    2. Output: a = g(z), with g being the activation function

    Keyword attributes:

    X
      The input vector containing examples organized in rows. The input
      matrix does **not** contain the bias term.

    Returns the outputs of the network for each row in X. Accumulates hidden
    layer outputs and activations (for backward step). At the end of this
    procedure:
    
    self.a
      Input, including the bias term for all layers. 1 example per row. Bias =
      first column.

    self.z
      Activations for every input X on given layer. z1 = a0 * w1
    """
    if self.has_bias:
      self.a = [numpy.hstack([numpy.ones((len(X),1)), X])]
    else:
      self.a = [X]

    self.z = []

    for w in self.weights[:-1]:
      self.z.append(numpy.dot(self.a[-1], w))
      self.a.append(self.hidden_activation(self.z[-1]))
      if self.has_bias:
        self.a[-1] = numpy.hstack([numpy.ones((len(self.a[-1]),1)), self.a[-1]])

    self.z.append(numpy.dot(self.a[-1], self.weights[-1]))
    self.a.append(self.output_activation(self.z[-1]))

    return self.a[-1]
