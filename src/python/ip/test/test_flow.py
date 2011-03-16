#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed  9 Mar 08:49:39 2011 

"""Tests our Optical Flow utilities going through some example data.
"""

import os, sys
import unittest
import torch

def load_gray(relative_filename):
  # Please note our PNG loader will always load in RGB, but since that is a
  # grayscaled version of the image, I just select one of the planes. 
  filename = os.path.join("data", "flow", relative_filename)
  array = torch.database.Array(filename)
  return array.get()[0,:,:] 

class FlowTest(unittest.TestCase):
  """Performs various combined optical flow tests."""
  
  def test01_VanillaHornAndSchunck(self):

    # Tests and examplifies usage of the vanilla HS algorithm

    # We create a new estimator specifying the alpha parameter (first value)
    # and the number of iterations to perform (second value).
    estimator = torch.ip.HornAndSchunckFlow(15, 100)

    # This will load the test images:
    i1 = load_gray(os.path.join("rubberwhale", "frame10_gray.png"))
    i2 = load_gray(os.path.join("rubberwhale", "frame11_gray.png"))

    # The OpticalFlow estimator always receives a blitz::Array<uint8_t,2> as
    # the image input. The output has the same rank and extents but is in
    # doubles.
    u = torch.core.array.float64_2()
    v = torch.core.array.float64_2()
    estimator(i1, i2, u, v)

    # Now we compare the output of the estimator with the existing computed
    # values.
    print u
    print v

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
