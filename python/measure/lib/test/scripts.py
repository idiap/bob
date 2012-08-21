#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 12:14:43 CEST 
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

"""Script tests for bob.measure
"""

import os
import unittest
import bob

DEV_SCORES = 'test-4col.txt'
TEST_SCORES = 'test-4col.txt'

class MeasureScriptTest(unittest.TestCase):

  def test01_compute_perf(self):

    from bob.measure.script.compute_perf import main
    cmdline = '--devel=%s --test=%s --self-test' % (DEV_SCORES, TEST_SCORES)
    self.assertEqual(main(cmdline.split()), 0)

  def test02_eval_threshold(self):

    from bob.measure.script.eval_threshold import main
    cmdline = '--scores=%s --self-test' % (DEV_SCORES,)
    self.assertEqual(main(cmdline.split()), 0)

  def test03_apply_threshold(self):

    from bob.measure.script.apply_threshold import main
    dev = bob.build.source_file('python/measure/data/test-4col.txt')
    self.assertTrue(os.path.exists(test))
    cmdline = '--scores=%s --self-test' % (TEST_SCORES,)
    self.assertEqual(main(cmdline.split()), 0)

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(MeasureScriptTest)
