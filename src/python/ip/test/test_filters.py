#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 12:33:13 CEST 

"""Test all ip image filters.
"""

import os, sys

def test_file(name):
  """Returns the path to the filename for this test."""
  return os.path.join("data", "filter", name)

# These are some global parameters for the test.
INPUT_IMAGE = test_file('image.ppm') #this a 100x100 pixel image of a face

import unittest
import torch

class FilterTest(unittest.TestCase):
  """Performs various combined filter tests."""
  """  
  def test01_crop(self):
    v = torch.ip.Image(1, 1, 3) 
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipCrop()
    self.assertEqual(f.setIOption('x', 50), True)
    self.assertEqual(f.setIOption('y', 50), True)
    self.assertEqual(f.setIOption('w', 20), True)
    self.assertEqual(f.setIOption('h', 20), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('cropped.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 3)
    self.assertTrue(reference.load(test_file('cropped.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test02_flip(self):
    v = torch.ip.Image(1, 1, 3) 
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipFlip()
    self.assertEqual(f.setBOption('vertical', True), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('flipped.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 3)
    self.assertTrue(reference.load(test_file('flipped.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))
  """

  def test03_histo(self):
    """
    v = torch.ip.Image(1, 1, 3) 
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipHisto()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    processed = f.getOutput(0)
    self.assertEqual(processed.getDatatype(), torch.core.Type.Int)
    #save_ref = torch.core.TensorFile() #use this to save a new reference
    #save_ref.openWrite(test_file('histo.tensorfile'), processed)
    #save_ref.save(processed)
    #save_ref.close()
    # compare to our model
    ref_file = torch.core.TensorFile()
    ref_file.openRead(test_file('histo.tensorfile'))
    reference = ref_file.load()
    for i in range(reference.size(0)):
      for j in range(reference.size(1)):
        self.assertEqual(processed.get(i,j), reference.get(i,j))
    ref_file.close()
    """

  def test04_histoequal(self):
    v = torch.ip.Image(1, 1, 1) #histo equalization only works on grayscaled
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipHistoEqual()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('histoequal.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('histoequal.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))
  
  def test05_ii(self):
    """
    v = torch.ip.Image(1, 1, 1) #only works on grayscaled
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipIntegral()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    processed = f.getOutput(0)
    self.assertEqual(processed.getDatatype(), torch.core.Type.Int)
    #save_ref = torch.core.TensorFile() #use this to save a new reference
    #save_ref.openWrite(test_file('integralimage.tensorfile'), processed)
    #save_ref.save(processed)
    #save_ref.close()
    # compare to our model
    ref_file = torch.core.TensorFile()
    ref_file.openRead(test_file('integralimage.tensorfile'))
    reference = ref_file.load()
    for i in range(reference.size(0)):
      for j in range(reference.size(1)):
        for k in range(reference.size(2)):
          self.assertEqual(processed.get(i, j, k), reference.get(i, j, k))
    ref_file.close()
    """

  def test06_MSRSQIGaussian(self):
    v = torch.ip.Image(1, 1, 1) 
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipMSRSQIGaussian()
    self.assertEqual(f.setIOption('RadiusX', 6), True)
    self.assertEqual(f.setIOption('RadiusY', 6), True)
    self.assertEqual(f.setDOption('Sigma', 3), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('msrsqigauss.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('msrsqigauss.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test07_MultiscaleRetinex(self):
    v = torch.ip.Image(1, 1, 1)
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipMultiscaleRetinex()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('multiscaleretinex.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('multiscaleretinex.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test08_relaxation(self):
    v = torch.ip.Image(1, 1, 1)
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipRelaxation()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('relaxation.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('relaxation.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  """
  def test09_rescaleGray(self):
    v_file = torch.core.TensorFile()
    v_file.openRead(test_file('integralimage.tensorfile'))
    v = v_file.load()
    v_file.close()
    f = torch.ip.ipRescaleGray()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('rescalegray.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('rescalegray.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test10_rotate(self):
    v = torch.ip.Image(1, 1, 1)
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipRotate()
    self.assertEqual(f.setIOption('centerx', 50), True)
    self.assertEqual(f.setIOption('centery', 50), True)
    self.assertEqual(f.setDOption('angle', 27.9), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('rotated.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('rotated.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test11_scaleYX(self):
    v = torch.ip.Image(1, 1, 1) #histo equalization only works on grayscaled
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipScaleYX()
    self.assertEqual(f.setIOption('width', 50), True)
    self.assertEqual(f.setIOption('height', 50), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('scaled.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('scaled.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))
  """

  def test12_selfQuotientImage(self):
    v = torch.ip.Image(1, 1, 1) #histo equalization only works on grayscaled
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipSelfQuotientImage()
    self.assertEqual(f.setIOption('s_nb', 3), True)
    self.assertEqual(f.setIOption('s_min', 2), True)
    self.assertEqual(f.setIOption('s_step', 5), True)
    self.assertEqual(f.setDOption('Sigma', 0.5), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('selfquotient.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('selfquotient.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  """
  def test13_shift(self):
    v = torch.ip.Image(1, 1, 1)
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipShift()
    self.assertEqual(f.setIOption('shiftx', 10), True)
    self.assertEqual(f.setIOption('shifty', 10), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('shift.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('shift.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))
  """

  def test14_smoothGaussian(self):
    v = torch.ip.Image(1, 1, 3)
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipSmoothGaussian()
    self.assertEqual(f.setIOption('RadiusX', 5), True)
    self.assertEqual(f.setIOption('RadiusY', 5), True)
    self.assertEqual(f.setDOption('Sigma', 5.0), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('smoothgauss.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 3)
    self.assertTrue(reference.load(test_file('smoothgauss.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test15_sobel(self):
    """
    v = torch.ip.Image(1, 1, 1)
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipSobel()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 3)
    outputs = []
    for z in range(3): outputs.append(f.getOutput(z))
    for t in outputs: self.assertEqual(t.getDatatype(), torch.core.Type.Int)
    #save_ref = torch.core.TensorFile() #use this to save a new reference
    #save_ref.openWrite(test_file('sobel.tensorfile'), outputs[0])
    #for t in outputs: save_ref.save(t)
    #save_ref.close()
    # compare to our model
    ref_file = torch.core.TensorFile()
    ref_file.openRead(test_file('sobel.tensorfile'))
    for t in outputs:
      reference = ref_file.load()
      for i in range(reference.size(0)):
        for j in range(reference.size(1)):
          for k in range(reference.size(2)):
            self.assertEqual(t.get(i, j, k), reference.get(i, j, k))
    ref_file.close()
    """

  def test16_TanTriggs(self):
    v = torch.ip.Image(1, 1, 1) #Tan/Triggs only work with grayscaled images
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipTanTriggs()
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('tantriggs.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('tantriggs.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

  def test17_vcycle(self):
    v = torch.ip.Image(1, 1, 1) #Tan/Triggs only work with grayscaled images
    self.assertTrue(v.load(INPUT_IMAGE))
    f = torch.ip.ipVcycle()
    self.assertEqual(f.setIOption('n_grids', 2), True)
    self.assertEqual(f.setIOption('type', 1), True)
    self.assertEqual(f.setDOption('lambda', 0.1), True)
    self.assertEqual(f.process(v), True)
    self.assertEqual(f.getNOutputs(), 1)
    self.assertEqual(f.getOutput(0).getDatatype(), torch.core.Type.Short)
    processed = torch.ip.Image(f.getOutput(0))
    #processed.save(test_file('vcycle.ppm')) #use this to save another reference image
    # compare to our model
    reference = torch.ip.Image(1, 1, 1)
    self.assertTrue(reference.load(test_file('vcycle.ppm')))
    for i in range(reference.width):
      for j in range(reference.height):
        for k in range(reference.nplanes):
          self.assertEqual(processed.get(j, i, k), reference.get(j, i, k))

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
