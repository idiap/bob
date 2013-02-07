#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST 

"""Tests various examples for bob.io
"""

import unittest
from ...test import utils

class ExampleTest(unittest.TestCase):

  @utils.ffmpeg_found()
  def test01_video2frame(self):

    movie = utils.datafile('test.mov')

    from ..example.video2frame import main
    cmdline = ['--self-test', movie]
    self.assertEqual(main(cmdline), 0)
