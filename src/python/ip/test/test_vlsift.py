#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey.ch>
# Mon  5 Dec 21:35:00 2011 

"""Tests our SIFT features extractor
"""

import os, sys
import unittest
import torch
import numpy

def equal(x, y, epsilon):
  return (abs(x - y) < epsilon)

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class VLSiftTest(unittest.TestCase):
  """Performs various tests"""

  def test01_VLSiftPython(self):
    img = torch.io.load(os.path.join("data", "sift", "vlimg_ref.pgm"))
    ref = torch.io.Arrayset(os.path.join("data", "sift", "vlsift_ref.hdf5"))
    mysift1 = torch.ip.VLSIFT(478,640, 3, 6, -1)
    out = mysift1(img)
    self.assertTrue(len(out) == len(ref))
    for i in range(len(out)):
      # Forces the cast in VLFeat sift main() function
      outi_uint8 = numpy.array( out[i], dtype='uint8') 
      out[i][4:132] = outi_uint8[4:132]
      self.assertTrue(equals(out[i],ref[i],1e-3))
    
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
