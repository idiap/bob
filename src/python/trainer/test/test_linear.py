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

    s = range(0,10)
    s[0] = torch.core.array.float64_1([2.5, 2.4],(2,))
    s[1] = torch.core.array.float64_1([0.5, 0.7],(2,))
    s[2] = torch.core.array.float64_1([2.2, 2.9],(2,))
    s[3] = torch.core.array.float64_1([1.9, 2.2],(2,))
    s[4] = torch.core.array.float64_1([3.1, 3.0],(2,))
    s[5] = torch.core.array.float64_1([2.3, 2.7],(2,))
    s[6] = torch.core.array.float64_1([2., 1.6],(2,))
    s[7] = torch.core.array.float64_1([1., 1.1],(2,))
    s[8] = torch.core.array.float64_1([1.5, 1.6],(2,))
    s[9] = torch.core.array.float64_1([1.1, 0.9],(2,))

    data = torch.database.Arrayset()
    for i in range(0,10): data.append(s[i])

    M = torch.machine.LinearMachine()
    T = torch.trainer.SVDPCATrainer()
    T.train(M, data)

    print mymachine.getEigenvalues()
    print mymachine.getEigenvectors()

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
