#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST 

"""Tests various examples for bob.db
"""

import unittest
import bob

class ExampleTest(unittest.TestCase):

  def test01_iris_lda(self):

    from bob.db.example.iris_lda import main
    cmdline = ['--self-test']
    self.assertEqual(main(cmdline), 0)

  def test02_iris_backprop(self):

    from bob.db.example.iris_backprop import main
    cmdline = ['--self-test']
    self.assertEqual(main(cmdline), 0)

  def test03_iris_lda(self):

    from bob.db.example.iris_rprop import main
    cmdline = ['--self-test']
    self.assertEqual(main(cmdline), 0)
