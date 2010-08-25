#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 12:33:13 CEST 

"""Test all ip image filters.
"""

import os, sys

# These are some global parameters for the test.
INPUT_IMAGE = 'image.ppm'

import unittest
import torch

class FilterTest(unittest.TestCase):
  """Performs various combined filter tests."""
  
  def test01_crop(self):
    v = torch.ip.Image(1, 1, 3) 
    v.load(INPUT_IMAGE)
    f = torch.ip.ipCrop()
    self.assertEqual(f.setIOption('x', 300), True)
    self.assertEqual(f.setIOption('y', 300), True)
    self.assertEqual(f.setIOption('w', 200), True)
    self.assertEqual(f.setIOption('h', 200), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    cropped = torch.ip.Image(f.getOutput(0))
    #cropped.save('cropped.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 3)
    reference.load('cropped.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(cropped.get(j, i, k), reference.get(j, i, k))

  def test02_flip(self):
    v = torch.ip.Image(1, 1, 3) 
    v.load(INPUT_IMAGE)
    f = torch.ip.ipFlip()
    self.assertEqual(f.setBOption('vertical', True), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    flipped = torch.ip.Image(f.getOutput(0))
    #flipped.save('flipped.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 3)
    reference.load('flipped.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(flipped.get(j, i, k), reference.get(j, i, k))
  
  def test03_histo(self):
    v = torch.ip.Image(1, 1, 3) 
    v.load(INPUT_IMAGE)
    f = torch.ip.ipHisto()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    histo = f.getOutput(0)
    self.assertEqual(histo.getDatatype(), torch.core.Type.Int)
    #save_ref = torch.core.TensorFile() #use this to save a new reference
    #save_ref.openWrite('histo.tensorfile', histo)
    #save_ref.save(histo)
    #save_ref.close()
    reference = torch.core.IntTensor()
    ref_file = torch.core.TensorFile()
    ref_file.openRead('histo.tensorfile')
    ref_file.load(reference)
    for i in range(reference.size(0)):
       self.assertEqual(reference.get(i), reference.get(i))
    ref_file.close()

if __name__ == '__main__':
  import sys
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
