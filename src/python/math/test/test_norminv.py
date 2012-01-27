#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <laurent.el-shafey@idiap.ch>

"""Tests for statistical methods
"""

import os, sys
import unittest
import bob
import numpy

class NorminvTest(unittest.TestCase):
  """Tests the norminv function of bob"""

  def test01_norminv(self):

    # Reference values
    sols_d05 = -1.64485362695
    sols_d50 = 0.
    sol_m2s4_d37 = 0.672586614252
    sol_m2s4_d48 = 1.799385666141

    # Values obtained with bob
    b_d05 = bob.math.normsinv(0.05)
    b_d50 = bob.math.normsinv(0.5)
    b_m2s4_d37 = bob.math.norminv(0.37, 2., 4.)
    b_m2s4_d48 = bob.math.norminv(0.48, 2., 4.)
  
    # Compare
    self.assertTrue( (abs(sols_d05 - b_d05) < 1e-6), True )
    self.assertTrue( (abs(sols_d50 - b_d50) < 1e-6), True )
    self.assertTrue( (abs(sol_m2s4_d37 - b_m2s4_d37) < 1e-6), True )
    self.assertTrue( (abs(sol_m2s4_d48 - b_m2s4_d48) < 1e-6), True )

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()

