#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 03 Nov 2010 12:24:27 CET 

"""Tests some functionality of the C++-Python exception bridge.
"""

import os, sys, unittest
import bob

class ExceptionTest(unittest.TestCase):
  """Performs various exception tests."""
  
  def test01_can_catch_from_cpp(self):
    self.assertRaises(RuntimeError, bob.core.throw_exception)
  """
  def test02_can_catch_from_another_module(self):
    self.assertRaises(RuntimeError, bob.ip.throw_exception)
  """

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
