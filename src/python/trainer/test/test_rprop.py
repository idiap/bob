#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 14 Jul 17:52:14 2011 

"""Tests for RProp MLP training.
"""

import os, sys
import unittest
import torch

class PythonRProp:
  """A simplified (and slower) version of RProp training written in python.
  
  This version of the code is probably easier to understand than the C++
  version. Both algorithms should, essentially, be the same, except for the
  performance for obvious reasons.
  """

  def __init__(self, train_biases=True):
    # Our state
    self.DW = None #delta matrixes for weights
    self.DB = None #delta matrixes for biases
    self.PDW = None #partial derivatives for weights
    self.PDB = None #partial derivatives for biases
    self.PPDW = None #previous partial derivatives for weights
    self.PPDB = None #previous partial derivatives for biases
    self.train_biases = train_biases

  def reset(self):
    """Resets our internal state"""

    self.DW = None #delta matrixes for weights
    self.DB = None #delta matrixes for biases
    self.PDW = None #partial derivatives for weights
    self.PDB = None #partial derivatives for biases
    self.PPDW = None #previous partial derivatives for weights
    self.PPDB = None #previous partial derivatives for biases

  def train(self, machine, input, target):

    def sign(x):
      """A handy sign function"""
      if (x == 0): return 0
      if (x < 0) : return -1
      return +1

    def logistic(x):
      return 1 / (1 + exp(-x))

    def logistic_bwd(x):
      return x * (1-x)

    def tanh(x):
      return x.tanh()

    def tanh_bwd(x):
      return (1 - x**2)

    def linear(x):
      return x

    def linear_bwd(x):
      return 1
    
    # some constants for RProp
    DELTA0 = 0.1
    DELTA_MIN = 1e-6
    DELTA_MAX = 50
    ETA_MINUS = 0.5
    ETA_PLUS = 1.2

    W = machine.weights #weights
    B = machine.biases #biases

    if machine.activation == torch.machine.Activation.TANH:
      forward = tanh
      backward = tanh_bwd
    elif machine.activation == torch.machine.Activation.LOG:
      forward = logistic
      backward = logistic_bwd
    elif machine.activation == torch.machine.Activation.LINEAR:
      forward = linear
      backward = linear_bwd
    else:
      raise RuntimeError, "Cannot deal with activation %s" % machine.activation
    
    #simulated bias input...
    BI = [torch.core.array.float64_1(input.extent(0)) for k in B]
    for k in BI: k.fill(1)

    #state
    if self.DW is None: #first run or just after a reset()
      self.DW = [k.empty_like() for k in W]
      for k in self.DW: k.fill(DELTA0)
      self.DB = [k.empty_like() for k in B]
      for k in self.DB: k.fill(DELTA0)
      self.PPDW = [k.empty_like() for k in W]
      for k in self.PPDW: k.fill(0)
      self.PPDB = [k.empty_like() for k in B]
      for k in self.PPDB: k.fill(0)

    # Instantiate partial outputs and errors
    O = [None for k in B]
    O.insert(0, input) # an extra slot for the input
    E = [None for k in B]

    # Feeds forward
    for k in range(len(W)):
      O[k+1] = torch.math.prod(O[k], W[k])
      for sample in range(O[k+1].extent(0)):
        O[k+1][sample,:] += B[k]
      O[k+1] = forward(O[k+1])

    # Feeds backward
    E[-1] = backward(O[-1]) * (O[-1] - target) #last layer
    for k in reversed(range(len(W)-1)): #for all remaining layers
      E[k] = backward(O[k+1]) * torch.math.prod(E[k+1], W[k+1].transpose(1,0))

    # Calculates partial derivatives, accumulate
    self.PDW = [torch.math.prod(O[k].transpose(1,0), E[k]) for k in range(len(W))]
    self.PDB = [torch.math.prod(BI[k], E[k]) for k in range(len(W))]

    # Updates weights and biases
    WUP = [i * j for (i,j) in zip(self.PPDW, self.PDW)]
    BUP = [i * j for (i,j) in zip(self.PPDB, self.PDB)]

    # Iterate over each weight and bias and see what to do:
    for k, up in enumerate(WUP):
      for i in range(up.extent(0)):
        for j in range(up.extent(1)):
          if up[i,j] > 0:
            self.DW[k][i,j] = min(self.DW[k][i,j]*ETA_PLUS, DELTA_MAX)
            W[k][i,j] -= sign(self.PDW[k][i,j]) * self.DW[k][i,j]
            self.PPDW[k][i,j] = self.PDW[k][i,j]
          elif up[i,j] < 0:
            self.DW[k][i,j] = max(self.DW[k][i,j]*ETA_MINUS, DELTA_MIN)
            self.PPDW[k][i,j] = 0
          elif up[i,j] == 0:
            W[k][i,j] -= sign(self.PDW[k][i,j]) * self.DW[k][i,j]
            self.PPDW[k][i,j] = self.PDW[k][i,j]
    machine.weights = W

    if self.train_biases:
      for k, up in enumerate(BUP):
        for i in range(up.extent(0)):
          if up[i] > 0:
            self.DB[k][i] = min(self.DB[k][i]*ETA_PLUS, DELTA_MAX)
            B[k][i] -= sign(self.PDB[k][i]) * self.DB[k][i]
            self.PPDB[k][i] = self.PDB[k][i]
          elif up[i] < 0:
            self.DB[k][i] = max(self.DB[k][i]*ETA_MINUS, DELTA_MIN)
            self.PPDB[k][i] = 0
          elif up[i] == 0:
            B[k][i] -= sign(self.PDB[k][i]) * self.DB[k][i]
            self.PPDB[k][i] = self.PDB[k][i]
      machine.biases = B

    else:
      machine.biases = 0

class RPropTest(unittest.TestCase):
  """Performs various RProp MLP training tests."""

  def test01_Initialization(self):

    # Initializes an MLPRPropTrainer and checks all seems consistent
    # with the proposed API.
    machine = torch.machine.MLP((4, 1))
    machine.activation = torch.machine.Activation.LINEAR
    B = 10
    trainer = torch.trainer.MLPRPropTrainer(machine, B)
    self.assertEqual( trainer.batchSize, B )
    self.assertTrue ( trainer.isCompatible(machine) )
    self.assertTrue ( trainer.trainBiases )

    machine = torch.machine.MLP((7, 2))
    self.assertFalse ( trainer.isCompatible(machine) )

    trainer.trainBiases = False
    self.assertFalse ( trainer.trainBiases )

  def test02_SingleLayerNoBiasControlled(self):

    # Trains a simple network with one single step, verifies
    # the training works as expected by calculating the same
    # as the trainer should do using python.
    machine = torch.machine.MLP((4, 1))
    machine.activation = torch.machine.Activation.LINEAR
    machine.biases = 0
    w0 = torch.core.array.array([[.1],[.2],[-.1],[-.05]])
    machine.weights = [w0]
    trainer = torch.trainer.MLPRPropTrainer(machine, 1)
    trainer.trainBiases = False
    d0 = torch.core.array.array([[1., 2., 0., 2.]])
    t0 = torch.core.array.array([[1.]])

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = torch.machine.MLP(machine) #a copy
    pytrainer.train(pymachine, d0, t0)

    # trains with our C++ implementation
    trainer.train_(machine, d0, t0)
    self.assertTrue( (pymachine.weights[0] == machine.weights[0]).all() )

    # a second passage
    d0 = torch.core.array.array([[4., 0., -3., 1.]])
    t0 = torch.core.array.array([[2.]])
    pytrainer.train(pymachine, d0, t0)
    trainer.train_(machine, d0, t0)
    self.assertTrue( (pymachine.weights[0] == machine.weights[0]).all() )

    # a third passage
    d0 = torch.core.array.array([[-0.5, -9.0, 2.0, 1.1]])
    t0 = torch.core.array.array([[3.]])
    pytrainer.train(pymachine, d0, t0)
    trainer.train_(machine, d0, t0)
    self.assertTrue( (pymachine.weights[0] == machine.weights[0]).all() )

  def test03_FisherNoBias(self):
    
    # Trains single layer MLP to discriminate the iris plants from
    # Fisher's paper. Checks we get a performance close to the one on
    # that paper.

    N = 60

    machine = torch.machine.MLP((4, 1))
    machine.activation = torch.machine.Activation.LINEAR
    machine.randomize()
    machine.biases = 0
    trainer = torch.trainer.MLPRPropTrainer(machine, N)
    trainer.trainBiases = False

    # A helper to select and shuffle the data
    targets = [ #we choose the approximate Fisher response!
        torch.core.array.array([-2.0]), #setosa
        torch.core.array.array([1.5]), #versicolor
        torch.core.array.array([0.5]), #virginica
        ]
    # Associate the data to targets, by setting the arrayset order explicetly
    data = torch.db.iris.data()
    datalist = [data['setosa'], data['versicolor'], data['virginica']]

    S = torch.trainer.DataShuffler(datalist, targets)

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = torch.machine.MLP(machine) #a copy

    # We now iterate for several steps, look for the convergence
    for k in range(100):
      input, target = S(N)
      pytrainer.train(pymachine, input, target)
      trainer.train_(machine, input, target)
      #print "[Python] MSE:", torch.measure.mse(pymachine(input), target).sqrt()
      #print "[C++] MSE:", torch.measure.mse(machine(input), target).sqrt()
      self.assertTrue( (pymachine.weights[0] == machine.weights[0]).all() )

  def test04_Fisher(self):
    
    # Trains single layer MLP to discriminate the iris plants from
    # Fisher's paper. Checks we get a performance close to the one on
    # that paper.

    N = 60

    machine = torch.machine.MLP((4, 1))
    machine.activation = torch.machine.Activation.LINEAR
    machine.randomize()
    trainer = torch.trainer.MLPRPropTrainer(machine, N)
    trainer.trainBiases = True

    # A helper to select and shuffle the data
    targets = [ #we choose the approximate Fisher response!
        torch.core.array.array([-2.0]), #setosa
        torch.core.array.array([1.5]), #versicolor
        torch.core.array.array([0.5]), #virginica
        ]
    # Associate the data to targets, by setting the arrayset order explicetly
    data = torch.db.iris.data()
    datalist = [data['setosa'], data['versicolor'], data['virginica']]

    S = torch.trainer.DataShuffler(datalist, targets)

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = torch.machine.MLP(machine) #a copy

    # We now iterate for several steps, look for the convergence
    for k in range(100):
      input, target = S(N)
      pytrainer.train(pymachine, input, target)
      trainer.train_(machine, input, target)
      #print "[Python] MSE:", torch.measure.mse(pymachine(input), target).sqrt()
      #print "[C++] MSE:", torch.measure.mse(machine(input), target).sqrt()
      self.assertTrue( (pymachine.weights[0] == machine.weights[0]).all() )
      self.assertTrue( (pymachine.biases[0] == machine.biases[0]).all() )

  def test05_FisherWithOneHiddenLayer(self):

    # Trains a multilayer biased MLP to perform discrimination on the Fisher
    # data set.

    N = 50

    machine = torch.machine.MLP((4, 4, 3))
    machine.activation = torch.machine.Activation.TANH
    machine.randomize()
    trainer = torch.trainer.MLPRPropTrainer(machine, N)
    trainer.trainBiases = True

    # A helper to select and shuffle the data
    targets = [ #we choose the approximate Fisher response!
        torch.core.array.array([+1., -1., -1.]), #setosa
        torch.core.array.array([-1., +1., -1.]), #versicolor
        torch.core.array.array([-1., -1., +1.]), #virginica
        ]
    # Associate the data to targets, by setting the arrayset order explicetly
    data = torch.db.iris.data()
    datalist = [data['setosa'], data['versicolor'], data['virginica']]

    S = torch.trainer.DataShuffler(datalist, targets)

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = torch.machine.MLP(machine) #a copy

    # We now iterate for several steps, look for the convergence
    for k in range(50):
      input, target = S(N)
      pytrainer.train(pymachine, input, target)
      trainer.train_(machine, input, target)
      #print "[Python] |RMSE|:", torch.math.norm(torch.measure.rmse(pymachine(input), target))
      #print "[C++] |RMSE|:", torch.math.norm(torch.measure.rmse(machine(input), target))
      for i, w in enumerate(pymachine.weights):
        self.assertTrue( (w == machine.weights[i]).all() )
      for i, b in enumerate(pymachine.biases):
        self.assertTrue( (b == machine.biases[i]).all() )

  def test06_FisherMultiLayer(self):

    # Trains a multilayer biased MLP to perform discrimination on the Fisher
    # data set.

    N = 50

    machine = torch.machine.MLP((4, 3, 3, 1))
    machine.activation = torch.machine.Activation.TANH
    machine.randomize()
    trainer = torch.trainer.MLPRPropTrainer(machine, N)
    trainer.trainBiases = True

    # A helper to select and shuffle the data
    targets = [ #we choose the approximate Fisher response!
        torch.core.array.array([-1.0]), #setosa
        torch.core.array.array([0.5]), #versicolor
        torch.core.array.array([+1.0]), #virginica
        ]
    # Associate the data to targets, by setting the arrayset order explicetly
    data = torch.db.iris.data()
    datalist = [data['setosa'], data['versicolor'], data['virginica']]

    S = torch.trainer.DataShuffler(datalist, targets)

    # trains in python first
    pytrainer = PythonRProp(train_biases=trainer.trainBiases)
    pymachine = torch.machine.MLP(machine) #a copy

    # We now iterate for several steps, look for the convergence
    for k in range(50):
      input, target = S(N)
      pytrainer.train(pymachine, input, target)
      trainer.train_(machine, input, target)
      #print "[Python] MSE:", torch.measure.mse(pymachine(input), target).sqrt()
      #print "[C++] MSE:", torch.measure.mse(machine(input), target).sqrt()
      for i, w in enumerate(pymachine.weights):
        self.assertTrue( (w == machine.weights[i]).all() )
      for i, b in enumerate(pymachine.biases):
        self.assertTrue( (b == machine.biases[i]).all() )

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()


