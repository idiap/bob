#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu Jan 27 10:47:36 2011 +0100
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
