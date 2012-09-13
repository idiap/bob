#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST 

"""Tests various examples for bob.io
"""

import unittest
import bob

class ExampleTest(unittest.TestCase):

  def test01_video2frame(self):

    import os

    from pkg_resources import resource_filename
    movie = resource_filename(__name__, os.path.join('data', 'test.mov'))

    from bob.io.example.video2frame import main
    cmdline = ['--self-test', movie]
    self.assertEqual(main(cmdline), 0)
