#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Apr 20 17:32:54 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Basic tests for the error measuring system of bob
"""

import os, sys
import unittest
import numpy
import bob
import pkg_resources


class LoadTest(unittest.TestCase):
  """Checks that the four-column and five-column load functions utils work."""

  def test01_load_4column(self):
    # tests that the bob.measure.load.four_column works as expected
    test_filename = bob.test.utils.datafile('test-4col.txt', 'bob.measure.test', 'data')

    lines = bob.measure.load.four_column(test_filename)
    self.assertEqual(len(lines), 910)
    for line in lines:
      self.assertEqual(len(line), 4)
      self.assertTrue(isinstance(line[3], float))

  def test02_load_5column(self):
    # tests that the bob.measure.load.five_column works as expected
    test_filename = bob.test.utils.datafile('test-5col.txt', 'bob.measure.test', 'data')

    lines = bob.measure.load.five_column(test_filename)
    self.assertEqual(len(lines), 910)
    for line in lines:
      self.assertEqual(len(line), 5)
      self.assertTrue(isinstance(line[4], float))

  def test03_tar(self):
    # tests that the tarfile interface for score files work
    test_filename = bob.test.utils.datafile('test-4col.txt', 'bob.measure.test', 'data')
    tar_filename = bob.test.utils.datafile('test-4col.tar.gz', 'bob.measure.test', 'data')

    test_lines = bob.measure.load.four_column(test_filename)
    tar_lines = bob.measure.load.four_column(tar_filename)

    self.assertEqual(len(test_lines), len(tar_lines))
    for i in range(len(test_lines)):
      self.assertEqual(len(test_lines[i]), len(tar_lines[i]))
      self.assertEqual(test_lines[i], tar_lines[i])
