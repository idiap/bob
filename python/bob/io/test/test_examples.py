#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:20:38 CEST

"""Tests various examples for bob.io
"""

from ...test import utils

@utils.ffmpeg_found()
def test_video2frame():

  movie = utils.datafile('test.mov', __name__)

  from ..example.video2frame import main
  cmdline = ['--self-test', movie]
  assert main(cmdline) == 0
