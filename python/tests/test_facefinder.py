#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 13 Jul 2010 10:18:55 CEST 

"""Runs some face finder tests
"""

INPUT_VIDEO = 'test.mov'
PARAMETERS = 'facefinder.multiscale.params'
CONTEXT_PARAMETERS = 'facefinder.track.context.params'

import unittest
import torch

class FaceFinderTest(unittest.TestCase):
  """Performs various tests for the Torch::FaceFinder object."""
 
  def test01_CanCreate(self):
    finder = torch.scanning.FaceFinder(PARAMETERS)

  def test02_CanFindOneFace(self):
    finder = torch.scanning.FaceFinder(PARAMETERS)
    v = torch.ip.Video(INPUT_VIDEO) 
    i = torch.ip.Image(1, 1, 1) #converts all to grayscale automagically!
    self.assertEqual(v.read(i), True)
    self.assertEqual(finder.process(i), True)
    patterns = finder.getPatterns()
    self.assertEqual(patterns.size(), 1)

  def test03_CanFindFacesInVideoSequence(self):
    finder = torch.scanning.FaceFinder(PARAMETERS)
    v = torch.ip.Video(INPUT_VIDEO) 
    i = torch.ip.Image(1, 1, 1) #converts all to grayscale automagically!
    n = 0
    for k in range(20):
      self.assertEqual(v.read(i), True)
      self.assertEqual(finder.process(i), True)
      patterns = finder.getPatterns()
      n += len(patterns)
    self.assertEqual(n > 15, True)

  def test04_CanFindOnROI(self):
    finder = torch.scanning.FaceFinder(PARAMETERS)
    v = torch.ip.Video(INPUT_VIDEO) 
    i = torch.ip.Image(1, 1, 1) #converts all to grayscale automagically!
    self.assertEqual(v.read(i), True)
    self.assertEqual(finder.process(i), True)
    patterns = finder.getPatterns()
    self.assertEqual(patterns.size(), 1)
    finder.getScanner().deleteAllROIs()
    self.assertEqual(finder.getScanner().addROIs(i, 0.3), True)
    self.assertEqual(finder.getScanner().getNoROIs(), 1)
    self.assertEqual(v.read(i), True)
    self.assertEqual(finder.process(i), True)
    patterns = finder.getPatterns()
    self.assertEqual(patterns.size(), 1)

  def test05_CanFindWithContextTracker(self):
    finder = torch.scanning.FaceFinder(CONTEXT_PARAMETERS)
    v = torch.ip.Video(INPUT_VIDEO) 
    i = torch.ip.Image(1, 1, 1) #converts all to grayscale automagically!
    self.assertEqual(v.read(i), True)
    self.assertEqual(finder.process(i), True)
    patterns = finder.getPatterns()
    self.assertEqual(patterns.size(), 1)
    self.assertNotEqual(finder.getScanner().tryGetTrackContextExplorer(), None)
    finder.getScanner().tryGetTrackContextExplorer().setSeedPatterns(patterns)
    self.assertEqual(v.read(i), True)
    self.assertEqual(finder.process(i), True)
    patterns = finder.getPatterns()
    self.assertEqual(patterns.size(), 1)

if __name__ == '__main__':
  import os, sys
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
