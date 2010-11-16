#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 12:33:13 CEST 

import os, sys
import unittest
import torch

def compare(v1, v2, width):
  return abs(v1-v2) <= width

class TransformTest(unittest.TestCase):
  """Performs for dct, dct2, fft, fft2 and their inverses"""
  
  def test_dct_1(self):

    # set up simple tensor (have to be 2d)
    t = torch.core.FloatTensor(2, 2)
    t.set(0, 0, 1.0)
    t.set(0, 1, 0.0)
    t.set(1, 0, 0.0)
    t.set(1, 1, 0.0)

    # process using DCT
    d = torch.sp.spDCT()
    d.process(t)
    self.assertEqual(d.getNOutputs(), 1)

    # get answer and compare to matlabs dct2 (warning do not use dct)
    tt = d.getOutput(0)

    self.assertTrue(compare(tt.get(0,0), 0.5, 1e-2))
    self.assertTrue(compare(tt.get(0,1), 0.5, 1e-2))
    self.assertTrue(compare(tt.get(1,0), 0.5, 1e-2))
    self.assertTrue(compare(tt.get(1,1), 0.5, 1e-2))
    
  def test_dct_2(self):

    # set up simple tensor (have to be 2d)
    t = torch.core.FloatTensor(4, 4)

    t.set(0, 0, 1.0)
    t.set(0, 1, 2.0)
    t.set(0, 2, 3.0)
    t.set(0, 3, 4.0)

    t.set(1, 0, 2.0)
    t.set(1, 1, 3.0)
    t.set(1, 2, 4.0)
    t.set(1, 3, 5.0)

    t.set(1, 0, 3.0)
    t.set(1, 1, 4.0)
    t.set(1, 2, 5.0)
    t.set(1, 3, 6.0)

    t.set(1, 0, 4.0)
    t.set(1, 1, 5.0)
    t.set(1, 2, 6.0)
    t.set(1, 3, 7.0)

    # process using DCT
    d = torch.sp.spDCT()
    d.process(t)
    self.assertEqual(d.getNOutputs(), 1)

    # get answer and compare to matlabs dct2 (warning do not use dct)
    tt = d.getOutput(0)

    self.assertTrue(compare(tt.get(0,0), 16.0000, 1e-3))
    self.assertTrue(compare(tt.get(0,1), -4.4609, 1e-3))
    self.assertTrue(compare(tt.get(0,2),  0.0000, 1e-3))
    self.assertTrue(compare(tt.get(0,3), -0.3170, 1e-3))

    self.assertTrue(compare(tt.get(1,0), -4.4609, 1e-3))
    self.assertTrue(compare(tt.get(1,1),  0.0000, 1e-3))
    self.assertTrue(compare(tt.get(1,2),  0.0000, 1e-3))
    self.assertTrue(compare(tt.get(1,3),  0.0000, 1e-3))

    self.assertTrue(compare(tt.get(2,0), 0.0000, 1e-3))
    self.assertTrue(compare(tt.get(2,1), 0.0000, 1e-3))
    self.assertTrue(compare(tt.get(2,2), 0.0000, 1e-3))
    self.assertTrue(compare(tt.get(2,3), 0.0000, 1e-3))

    self.assertTrue(compare(tt.get(3,0), -0.3170, 1e-3))
    self.assertTrue(compare(tt.get(3,1),  0.0000, 1e-3))
    self.assertTrue(compare(tt.get(3,2),  0.0000, 1e-3))
    self.assertTrue(compare(tt.get(3,3),  0.0000, 1e-3))

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
