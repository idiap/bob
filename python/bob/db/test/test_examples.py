#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST 

"""Tests various examples for bob.db
"""

import unittest
from ...test import utils

class ExampleTest(unittest.TestCase):

  def test01_iris_lda(self):

    from ..example.iris_lda import main
    cmdline = ['--self-test']
    self.assertEqual(main(cmdline), 0)

  @utils.ffmpeg_found()
  def test02_iris_backprop(self):

    from ..example.iris_backprop import main
    cmdline = ['--self-test']
    self.assertEqual(main(cmdline), 0)

  @utils.ffmpeg_found()
  def test03_iris_rprop(self):

    from ..example.iris_rprop import main
    cmdline = ['--self-test']
    self.assertEqual(main(cmdline), 0)
