#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 03 Nov 2010 12:24:27 CET 

"""Tests some functionality of the C++-Python exception bridge.
"""

import os, sys, unittest
import torch

class ExceptionTest(unittest.TestCase):
  """Performs various exception tests."""
  
  def test01_can_catch_from_cpp(self):
    self.assertRaises(torch.core.Exception, torch.core.throw_exception)
  """
  def test02_can_catch_from_another_module(self):
    self.assertRaises(torch.core.Exception, torch.ip.throw_exception)
  """

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
