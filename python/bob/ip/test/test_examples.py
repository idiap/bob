#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST 

"""Tests various examples for bob.ip
"""

import unittest
from ...test import utils
from ...io import test as iotest

class ExampleTest(unittest.TestCase):

  @utils.ffmpeg_found()
  def test01_optflow_hs(self):

    movie = utils.datafile('test.mov', iotest)
    from ..example.optflow_hs import main
    cmdline = ['--self-test', movie, '__ignored__']
    self.assertEqual(main(cmdline), 0)
