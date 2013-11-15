#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu Jan 27 10:47:36 2011 +0100
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests some functionality of the C++-Python exception bridge.
"""

import unittest
import bob

class ExceptionTest(unittest.TestCase):
  """Performs various exception tests."""
  
  def test01_can_catch_from_cpp(self):
    self.assertRaises(RuntimeError, bob.core.throw_exception)
  """
  def test02_can_catch_from_another_module(self):
    self.assertRaises(RuntimeError, bob.ip.throw_exception)
  """
