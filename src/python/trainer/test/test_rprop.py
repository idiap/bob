#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 14 Jul 17:52:14 2011 

"""Tests for RProp MLP training.
"""

import os, sys
import unittest
import torch
import numpy

class RPropTest(unittest.TestCase):
  """Performs various RProp MLP training tests."""

  def test01_Initialization(self):

    # Initializes an MLPRPropTrainer and checks all seems consistent
    # with the proposed API.
    machine = torch.machine.MLP((4, 1))
    B = 10
    trainer = torch.trainer.MLPRPropTrainer(machine, B)
    self.assertEqual( trainer.batchSize, B )
    self.assertTrue ( trainer.isCompatible(machine) )

    machine = torch.machine.MLP((7, 2))
    self.assertFalse ( trainer.isCompatible(machine) )

  def test02_SingleLayerSingleStep(self):

    # Trains a simple network with one single step, verifies
    # the training works as expected by calculating the same
    # as the trainer should do using numpy.
    machine = torch.machine.MLP((4, 1))
    trainer = torch.trainer.MLPRPropTrainer(machine, 10)

  def xtest03_Fisher(self):
    
    # Trains single layer MLP to discriminate the iris plants from
    # Fisher's paper. Checks we get a performance close to the one on
    # that paper.
    machine = torch.machine.MLP((4, 1))
    machine.randomize()
    trainer = torch.trainer.MLPRPropTrainer(machine, 10)
    
    pass

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


