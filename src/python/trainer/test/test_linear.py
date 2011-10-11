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


class TrainerTest(unittest.TestCase):
  """Performs various trainer tests."""
  
  def test01_pca_via_svd(self):

    # Tests our SVD/PCA extractor.
    data = torch.io.Arrayset()
    data.append(torch.core.array.float64_1([2.5, 2.4],(2,)))
    data.append(torch.core.array.float64_1([0.5, 0.7],(2,)))
    data.append(torch.core.array.float64_1([2.2, 2.9],(2,)))
    data.append(torch.core.array.float64_1([1.9, 2.2],(2,)))
    data.append(torch.core.array.float64_1([3.1, 3.0],(2,)))
    data.append(torch.core.array.float64_1([2.3, 2.7],(2,)))
    data.append(torch.core.array.float64_1([2., 1.6],(2,)))
    data.append(torch.core.array.float64_1([1., 1.1],(2,)))
    data.append(torch.core.array.float64_1([1.5, 1.6],(2,)))
    data.append(torch.core.array.float64_1([1.1, 0.9],(2,)))

    # Expected results
    eig_val_correct = torch.core.array.array([1.28402771, 0.0490834], 'float64')
    eig_vec_correct = torch.core.array.array([[-0.6778734, -0.73517866], [-0.73517866, 0.6778734]], 'float64')

    T = torch.trainer.SVDPCATrainer()
    machine, eig_vals = T.train(data)

    # Makes sure results are good
    self.assertTrue( ((machine.weights - eig_vec_correct) < 1e-6).all() )
    self.assertTrue( ((eig_vals - eig_val_correct) < 1e-6).all() )

  def test02_fisher_lda(self):

    # Tests our Fisher/LDA trainer for linear machines for a simple 2-class
    # "fake" problem:
    data = [torch.io.Arrayset(), torch.io.Arrayset()]
    data[0].append(torch.core.array.float64_1([2.5, 2.4],(2,)))
    data[0].append(torch.core.array.float64_1([2.2, 2.9],(2,)))
    data[0].append(torch.core.array.float64_1([1.9, 2.2],(2,)))
    data[0].append(torch.core.array.float64_1([3.1, 3.0],(2,)))
    data[0].append(torch.core.array.float64_1([2.3, 2.7],(2,)))
    data[1].append(torch.core.array.float64_1([0.5, 0.7],(2,)))
    data[1].append(torch.core.array.float64_1([2., 1.6],(2,)))
    data[1].append(torch.core.array.float64_1([1., 1.1],(2,)))
    data[1].append(torch.core.array.float64_1([1.5, 1.6],(2,)))
    data[1].append(torch.core.array.float64_1([1.1, 0.9],(2,)))

    # Expected results
    exp_trans_data = [
        [1.0019, 3.1205, 0.9405, 2.4962, 2.2949], 
        [-2.9042, -1.3179, -2.0172, -0.7720, -2.8428]
        ]
    exp_mean = torch.core.array.array([1.8100, 1.9100], 'float64')
    exp_val = torch.core.array.array([24.27536526], 'float64')
    exp_mach = torch.core.array.array([[-0.291529], [0.956562]], 'float64')

    T = torch.trainer.FisherLDATrainer()
    machine, eig_vals = T.train(data)

    # Makes sure results are good
    self.assertTrue( ((machine.input_subtract - exp_mean) < 1e-6).all() )
    self.assertTrue( ((machine.weights - exp_mach) < 1e-6).all() )
    self.assertTrue( ((eig_vals - exp_val) < 1e-6).all() )

  def test03_ppca(self):

    # Tests our Probabilistic PCA trainer for linear machines for a simple 
    # problem:
    ar=torch.io.Arrayset()
    ar.append(torch.core.array.float64_1([1,2,3], (3,)))
    ar.append(torch.core.array.float64_1([2,4,19], (3,)))
    ar.append(torch.core.array.float64_1([3,6,5], (3,)))
    ar.append(torch.core.array.float64_1([4,8,13], (3,)))
    
    # Expected llh 1 and 2 (Reference values)
    exp_llh1 =  -32.8443
    exp_llh2 =  -30.8559
   
    # Do two iterations of EM to check the training procedure 
    T = torch.trainer.EMPCATrainer(2)
    m = torch.machine.LinearMachine()
    # Initialization of the trainer
    T.initialization(m, ar)
    # Sets ('random') initialization values for test purposes
    w_init = torch.core.array.float64_2([1.62945, 0.270954, 1.81158, 1.67002, 0.253974, 1.93774], (3,2))
    sigma2_init = 1.82675
    m.weights = w_init
    T.sigma2 = sigma2_init
    # Checks that the log likehood matches the reference one
    # This should be sufficient to check everything as it requires to use
    # the new value of W and sigma2 
    # This does an E-Step, M-Step, computes the likelihood, and compares it to
    # the reference value obtained using matlab
    T.eStep(m, ar)
    T.mStep(m, ar)
    llh1 = T.computeLikelihood(m, ar)
    self.assertTrue( abs(exp_llh1 - llh1) < 2e-4)
    T.eStep(m, ar)
    T.mStep(m, ar)
    llh2 = T.computeLikelihood(m, ar)
    self.assertTrue( abs(exp_llh2 - llh2) < 2e-4)

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
