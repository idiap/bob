#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST 

"""Tests various examples for bob.ip
"""

import unittest
import bob

class ExampleTest(unittest.TestCase):

  def test01_optflow_hs(self):

    import os

    from pkg_resources import resource_filename
    movie = resource_filename(bob.io.__name__, os.path.join('test', 'data', 'test.mov'))

    from bob.ip.example.optflow_hs import main
    cmdline = ['--self-test', movie, '__ignored__']
    self.assertEqual(main(cmdline), 0)
