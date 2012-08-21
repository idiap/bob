#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 21 Aug 2012 13:30:54 CEST 

"""Test scripts in bob.visioner
"""

import os
import unittest
import bob

MOVIE = 'test.mov'
IMAGE = '../../ip/data/faceextract/test-faces.jpg'

class VisionerScriptTest(unittest.TestCase):

  def test01_face_detect(self):
   
    # sanity checks
    self.assertTrue(os.path.exists(MOVIE))

    from bob.visioner.script.facebox import main
    cmdline = '%s --self-test=1' % (MOVIE)
    self.assertEqual(main(cmdline.split()), 0)

  def test02_face_detect(self):
   
    # sanity checks
    self.assertTrue(os.path.exists(IMAGE))

    from bob.visioner.script.facebox import main
    cmdline = '%s --self-test=2' % (IMAGE)
    self.assertEqual(main(cmdline.split()), 0)

  def test03_keypoint_localization(self):
   
    # sanity checks
    self.assertTrue(os.path.exists(MOVIE))

    from bob.visioner.script.facepoints import main
    cmdline = '%s --self-test=1' % (MOVIE)
    self.assertEqual(main(cmdline.split()), 0)

  def test04_keypoint_localization(self):
   
    # sanity checks
    self.assertTrue(os.path.exists(IMAGE))

    from bob.visioner.script.facepoints import main
    cmdline = '%s --self-test=2' % (IMAGE)
    self.assertEqual(main(cmdline.split()), 0)

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(VisionerScriptTest)
