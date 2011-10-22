#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri 14 Oct 2011

"""Tests PLDA trainer
"""

import os, sys
import unittest
import torch
import random

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class PLDAMachineTest(unittest.TestCase):
  """Performs various PLDA machine tests."""
  
  def test01_plda_machine(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    D = 7
    nf = 2
    ng = 3
    # Values for F, G and sigma
    G=torch.core.array.float64_2([-1.1424, -0.5044, -0.1917,
                                       -0.6249,  0.1021, -0.8658,
                                       -1.1687,  1.1963,  0.1807,
                                        0.3926,  0.1203,  1.2665,
                                        1.3018, -1.0368, -0.2512,
                                       -0.5936, -0.8571, -0.2046,
                                        0.4364, -0.1699, -2.2015], (D,ng))
    # F <-> PCA on G
    F=torch.core.array.float64_2([-0.054222647972093, -0.000000000783146, 
                                        0.596449127693018,  0.000000006265167, 
                                        0.298224563846509,  0.000000003132583, 
                                        0.447336845769764,  0.000000009397750, 
                                       -0.108445295944185, -0.000000001566292, 
                                       -0.501559493741856, -0.000000006265167, 
                                       -0.298224563846509, -0.000000003132583], (D,nf))
    sigma = torch.core.array.float64_1((D,))
    sigma.fill(0.01)
    # Defines reference results based on matlab
    alpha_ref = torch.core.array.float64_2([ 0.002189051545735,  0.001127099941432, -0.000145483208153,
                                             0.001127099941432,  0.003549267943741, -0.000552001405453,
                                            -0.000145483208153, -0.000552001405453,  0.001440505362615], (ng,ng))
    beta_ref  = torch.core.array.float64_2([
      50.587191765140361, -14.512478352504877,  -0.294799164567830,  13.382002504394316,  
       9.202063877660278, -43.182264846086497,  11.932345916716455,
     -14.512478352504878,  82.320149045633045, -12.605578822979698,  19.618675892079366,
      13.033691341150439,  -8.004874490989799, -21.547363307109187,
      -0.294799164567832, -12.605578822979696,  52.123885798398241,   4.363739008635009,
      44.847177605628545,  16.438137537463710,   5.137421840557050,
      13.382002504394316,  19.618675892079366,   4.363739008635011,  75.070401560513488,
      -4.515472972526140,   9.752862741017488,  34.196127678931106,
       9.202063877660285,  13.033691341150439,  44.847177605628552,  -4.515472972526142,
      56.189416227691098,  -7.536676357632515, -10.555735414707383,
     -43.182264846086497,  -8.004874490989799,  16.438137537463703,   9.752862741017490,
      -7.536676357632518,  56.430571485722126,   9.471758169835317,
      11.932345916716461, -21.547363307109187,   5.137421840557051,  34.196127678931099,
     -10.555735414707385,   9.471758169835320,  27.996266602110637], (D,D))
    gamma3_ref = torch.core.array.float64_2([ 0.005318799462241, -0.000000012993151,
                                             -0.000000012993151,  0.999999999999996], (nf,nf))

    # Defines base machine
    m = torch.machine.PLDABaseMachine(D,nf,ng)
    # Sets the current F, G and sigma 
    # WARNING: order does matter, as this implies some precomputations
    m.sigma = sigma
    m.G = G
    m.F = F
    gamma3 = torch.core.array.float64_2((nf,nf))
    m.computeGamma(3, gamma3)

    # Compares precomputed values to matlab reference
    self.assertTrue(equals(m.alpha, alpha_ref, 1e-10))
    self.assertTrue(equals(m.beta, beta_ref, 1e-10))
    self.assertTrue(equals(gamma3, gamma3_ref, 1e-10))


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
