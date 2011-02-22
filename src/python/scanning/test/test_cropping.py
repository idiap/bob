#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 16 Jul 2010 09:30:53 CEST 

"""Tests cropping features
"""

import os, sys
import unittest
import torch

INPUT_VIDEO = '../../../ip/test/data/video/test.mov'
FACE_FINDER_PARAMETERS = 'facefinder.multiscale.params'
GEOMNORM_PARAMETERS = 'geom.norm-64x80.cfg'

class CroppingTest(unittest.TestCase):
  """Performs various tests for the Torch::ipGeomNorm object."""
 
  def test01_CanCropOne(self):
    finder = torch.scanning.FaceFinder(FACE_FINDER_PARAMETERS)
    v = torch.ip.Video(INPUT_VIDEO)
    i = torch.ip.Image(1, 1, 1) #converts all to grayscale automagically!
    self.assertEqual(v.read(i), True)
    self.assertEqual(finder.process(i), True)
    patterns = finder.getPatterns()
    self.assertEqual(patterns.size(), 1)
    geom_norm = torch.scanning.ipGeomNorm(GEOMNORM_PARAMETERS)
    gt_file = torch.scanning.BoundingBoxGTFile() #all in memory!
    self.assertEqual(geom_norm.setGTFile(gt_file), True)
    for index, k in enumerate(patterns): 
      gt_file.load(k)
      self.assertEqual(geom_norm.process(i), True)
      oi = geom_norm.getOutputImage(index)
      self.assertEqual(oi.width, 64)
      self.assertEqual(oi.height, 80)
      self.assertEqual(oi.nplanes, 1)

  def test02_CanCropMany(self):
    finder = torch.scanning.FaceFinder(FACE_FINDER_PARAMETERS)
    v = torch.ip.Video(INPUT_VIDEO)
    i = torch.ip.Image(1, 1, 1) #converts all to grayscale automagically!
    for frame in range(50):
      self.assertEqual(v.read(i), True)
      self.assertEqual(finder.process(i), True)
      patterns = finder.getPatterns()
      if len(patterns) == 0:
        #print 'Skipping frame %d' % frame
        continue
      self.assertEqual(patterns.size(), 1)
      geom_norm = torch.scanning.ipGeomNorm(GEOMNORM_PARAMETERS)
      gt_file = torch.scanning.BoundingBoxGTFile() #all in memory!
      self.assertEqual(geom_norm.setGTFile(gt_file), True)
      for index, k in enumerate(patterns): 
        gt_file.load(k)
        self.assertEqual(geom_norm.process(i), True)
        oi = geom_norm.getOutputImage(index)
        self.assertEqual(oi.width, 64)
        self.assertEqual(oi.height, 80)
        self.assertEqual(oi.nplanes, 1)

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  os.chdir('data')
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
