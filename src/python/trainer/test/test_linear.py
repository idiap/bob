#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 10 Jun 2011 16:24:13 CEST 

"""Test trainer package
"""

import os, sys
import unittest
import torch
import random
import numpy

class TrainerTest(unittest.TestCase):
  """Performs various trainer tests."""
  
  def test01_pca_via_svd(self):

    # Tests our SVD/PCA extractor.
    data = torch.io.Arrayset()
    data.append(numpy.array([2.5, 2.4]))
    data.append(numpy.array([0.5, 0.7]))
    data.append(numpy.array([2.2, 2.9]))
    data.append(numpy.array([1.9, 2.2]))
    data.append(numpy.array([3.1, 3.0]))
    data.append(numpy.array([2.3, 2.7]))
    data.append(numpy.array([2., 1.6]))
    data.append(numpy.array([1., 1.1]))
    data.append(numpy.array([1.5, 1.6]))
    data.append(numpy.array([1.1, 0.9]))

    # Expected results
    eig_val_correct = numpy.array([1.28402771, 0.0490834], 'float64')
    eig_vec_correct = numpy.array([[-0.6778734, -0.73517866], [-0.73517866, 0.6778734]], 'float64')

    T = torch.trainer.SVDPCATrainer()
    machine, eig_vals = T.train(data)

    # Makes sure results are good
    self.assertTrue( ((machine.weights - eig_vec_correct) < 1e-6).all() )
    self.assertTrue( ((eig_vals - eig_val_correct) < 1e-6).all() )

  def test02_fisher_lda(self):

    # Tests our Fisher/LDA trainer for linear machines for a simple 2-class
    # "fake" problem:
    data = [torch.io.Arrayset(), torch.io.Arrayset()]
    data[0].append(numpy.array([2.5, 2.4]))
    data[0].append(numpy.array([2.2, 2.9]))
    data[0].append(numpy.array([1.9, 2.2]))
    data[0].append(numpy.array([3.1, 3.0]))
    data[0].append(numpy.array([2.3, 2.7]))
    data[1].append(numpy.array([0.5, 0.7]))
    data[1].append(numpy.array([2., 1.6]))
    data[1].append(numpy.array([1., 1.1]))
    data[1].append(numpy.array([1.5, 1.6]))
    data[1].append(numpy.array([1.1, 0.9]))

    # Expected results
    exp_trans_data = [
        [1.0019, 3.1205, 0.9405, 2.4962, 2.2949], 
        [-2.9042, -1.3179, -2.0172, -0.7720, -2.8428]
        ]
    exp_mean = numpy.array([1.8100, 1.9100])
    exp_val = numpy.array([24.27536526])
    exp_mach = numpy.array([[-0.291529], [0.956562]])

    T = torch.trainer.FisherLDATrainer()
    machine, eig_vals = T.train(data)

    # Makes sure results are good
    self.assertTrue( ((machine.input_subtract - exp_mean) < 1e-6).all() )
    self.assertTrue( ((machine.weights - exp_mach) < 1e-6).all() )
    self.assertTrue( ((eig_vals - exp_val) < 1e-6).all() )

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
