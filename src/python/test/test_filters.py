#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 12:33:13 CEST 

"""Test all ip image filters.
"""

import os, sys

# These are some global parameters for the test.
INPUT_IMAGE = 'image.ppm' #this a 100x100 pixel image of a face

import unittest
import torch

class FilterTest(unittest.TestCase):
  """Performs various combined filter tests."""
  
  def test01_crop(self):
    v = torch.ip.Image(1, 1, 3) 
    v.load(INPUT_IMAGE)
    f = torch.ip.ipCrop()
    self.assertEqual(f.setIOption('x', 50), True)
    self.assertEqual(f.setIOption('y', 50), True)
    self.assertEqual(f.setIOption('w', 20), True)
    self.assertEqual(f.setIOption('h', 20), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('cropped.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 3)
    reference.load('cropped.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test02_flip(self):
    v = torch.ip.Image(1, 1, 3) 
    v.load(INPUT_IMAGE)
    f = torch.ip.ipFlip()
    self.assertEqual(f.setBOption('vertical', True), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('flipped.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 3)
    reference.load('flipped.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))
  
  def test03_histo(self):
    v = torch.ip.Image(1, 1, 3) 
    v.load(INPUT_IMAGE)
    f = torch.ip.ipHisto()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    processed = f.getOutput(0)
    self.assertEqual(processed.getDatatype(), torch.core.Type.Int)
    #save_ref = torch.core.TensorFile() #use this to save a new reference
    #save_ref.openWrite('histo.tensorfile', processed)
    #save_ref.save(processed)
    #save_ref.close()
    # compare to our model
    reference = torch.core.IntTensor()
    ref_file = torch.core.TensorFile()
    ref_file.openRead('histo.tensorfile')
    ref_file.load(reference)
    for i in range(reference.size(0)):
       self.assertEqual(reference.get(i), reference.get(i))
    ref_file.close()

  def test04_histoequal(self):
    v = torch.ip.Image(1, 1, 1) #histo equalization only works on grayscaled
    v.load(INPUT_IMAGE)
    f = torch.ip.ipHistoEqual()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('histoequal.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    reference.load('histoequal.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))
  
  def test05_ii(self):
    v = torch.ip.Image(1, 1, 1) #only works on grayscaled
    v.load(INPUT_IMAGE)
    f = torch.ip.ipIntegral()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    processed = f.getOutput(0)
    self.assertEqual(processed.getDatatype(), torch.core.Type.Int)
    #save_ref = torch.core.TensorFile() #use this to save a new reference
    #save_ref.openWrite('integralimage.tensorfile', processed)
    #save_ref.save(processed)
    #save_ref.close()
    # compare to our model
    reference = torch.core.IntTensor()
    ref_file = torch.core.TensorFile()
    ref_file.openRead('integralimage.tensorfile')
    ref_file.load(reference)
    for i in range(reference.size(0)):
      for j in range(reference.size(1)):
        for k in range(reference.size(2)):
          self.assertEqual(reference.get(i, j, k), reference.get(i, j, k))
    ref_file.close()

  def test06_MSRSQIGaussian(self):
    v = torch.ip.Image(1, 1, 1) 
    v.load(INPUT_IMAGE)
    f = torch.ip.ipMSRSQIGaussian()
    self.assertEqual(f.setIOption('RadiusX', 6), True)
    self.assertEqual(f.setIOption('RadiusY', 6), True)
    self.assertEqual(f.setDOption('Sigma', 3), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('msrsqigauss.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    reference.load('msrsqigauss.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test07_MultiscaleRetinex(self):
    v = torch.ip.Image(1, 1, 1)
    v.load(INPUT_IMAGE)
    f = torch.ip.ipMultiscaleRetinex()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('multiscaleretinex.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    reference.load('multiscaleretinex.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test08_relaxation(self):
    v = torch.ip.Image(1, 1, 1)
    v.load(INPUT_IMAGE)
    f = torch.ip.ipRelaxation()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('relaxation.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    reference.load('relaxation.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test09_rescaleGray(self):
    v_file = torch.core.TensorFile()
    v_file.openRead('integralimage.tensorfile')
    v = v_file.load()
    v_file.close()
    f = torch.ip.ipRescaleGray()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('rescalegray.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    reference.load('rescalegray.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test10_rotate(self):
    v = torch.ip.Image(1, 1, 1)
    v.load(INPUT_IMAGE)
    f = torch.ip.ipRotate()
    self.assertEqual(f.setIOption('centerx', 50), True)
    self.assertEqual(f.setIOption('centery', 50), True)
    self.assertEqual(f.setDOption('angle', 27.9), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('rotated.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    reference.load('rotated.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test11_scaleYX(self):
    v = torch.ip.Image(1, 1, 1) #histo equalization only works on grayscaled
    v.load(INPUT_IMAGE)
    f = torch.ip.ipScaleYX()
    self.assertEqual(f.setIOption('width', 50), True)
    self.assertEqual(f.setIOption('height', 50), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('scaled.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    reference.load('scaled.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test12_selfQuotientImage(self):
    v = torch.ip.Image(1, 1, 1) #histo equalization only works on grayscaled
    v.load(INPUT_IMAGE)
    f = torch.ip.ipSelfQuotientImage()
    self.assertEqual(f.setIOption('s_nb', 3), True)
    self.assertEqual(f.setIOption('s_min', 2), True)
    self.assertEqual(f.setIOption('s_step', 5), True)
    self.assertEqual(f.setDOption('Sigma', 0.5), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save('selfquotient.ppm') #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    reference.load('selfquotient.ppm')
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))
    

if __name__ == '__main__':
  import sys
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
