#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST

"""Tests various examples for bob.ip
"""

from ...test import utils
from ...io import test as iotest

@utils.ffmpeg_found()
def test01_optflow_hs():
  movie = utils.datafile('test.mov', iotest.__name__)
  from ..example.optflow_hs import main
  cmdline = ['--self-test', movie, '__ignored__']
  assert main(cmdline) == 0
