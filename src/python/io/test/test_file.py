#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Sat 19 Feb 06:58:47 2011 

"""A combined test for all built-in types of Array/Arrayset/File interaction in
python.
"""

import os, sys
import unittest
import tempfile
import torch
import numpy
import random

DEFAULT_EXTENSION = '.hdf5' # define here the codec you trust

# This test implements a generalized framework for testing codecs. It
# loads files in the codec native format, convert into torch native binary
# format and back, comparing the outcomes at every stage. We believe in the
# quality of the binary codec because that is covered in other tests.

def tempname(suffix, prefix='torchtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

def testcase_transcode(self, filename):
  """Runs a complete transcoding test, to and from the binary format."""

  tmpname = tempname(os.path.splitext(filename)[1])

  try:
    # transcode from test format into the test format -- test array access modes
    orig_data = torch.io.open(filename, 'r').read()
    torch.io.open(tmpname, 'w').write(orig_data)
    rewritten_data = torch.io.open(tmpname, 'r').read()

    self.assertTrue( numpy.array_equal(orig_data, rewritten_data) )

    # transcode to test format -- test arrayset access modes
    trans_file = torch.io.open(tmpname, 'w')
    index = [slice(orig_data.shape[k]) for k in range(len(orig_data.shape))]
    for k in range(orig_data.shape[0]):
      index[0] = k
      trans_file.append(orig_data[index]) #slice from first dimension
    del trans_file

    rewritten_file = torch.io.open(tmpname, 'r')

    for k in range(orig_data.shape[0]):
      rewritten_data = rewritten_file.read(k)
      index[0] = k
      self.assertTrue( numpy.array_equal(orig_data[index], rewritten_data) )

  finally:
    # And we erase both files after this
    if os.path.exists(tmpname): os.unlink(tmpname)

# We attach the transcoding method to the testcase class, so we can use its
# assertions
unittest.TestCase.transcode = testcase_transcode

def testcase_array_readwrite(self, extension, arr):
  """Runs a read/write verify step using the given numpy data"""
  tmpname = tempname(extension)
  try:
    f = torch.io.open(tmpname, 'w')
    f.write(arr)
    del f
    f = torch.io.open(tmpname, 'r')
    reloaded = f.read() #read the contents
    self.assertTrue(numpy.array_equal(arr, reloaded))
  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)

# And we attach...
unittest.TestCase.array_readwrite = testcase_array_readwrite

def testcase_arrayset_readwrite(self, extension, arrays):
  """Runs a read/write verify step using the given numpy data"""
  tmpname = tempname(extension)
  try:
    f = torch.io.open(tmpname, 'w')
    for k in arrays: 
      f.append(k)
    del f
    f = torch.io.open(tmpname, 'r')
    for k, array in enumerate(arrays):
      reloaded = f.read(k) #read the contents
      self.assertTrue(numpy.array_equal(array, reloaded))
  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)

# And we attach...
unittest.TestCase.arrayset_readwrite = testcase_arrayset_readwrite

class FileTest(unittest.TestCase):
  """Performs various tests for the Torch::io::*File types."""

  def test00_hdf5(self):

    # array writing tests
    a1 = numpy.random.normal(size=(2,3)).astype('float32')
    a2 = numpy.random.normal(size=(2,3,4)).astype('float64')
    a3 = numpy.random.normal(size=(2,3,4,5)).astype('complex128')
    a4 = (10 * numpy.random.normal(size=(3,3))).astype('uint64')

    self.array_readwrite('.hdf5', a1) # extensions: .hdf5 or .h5
    self.array_readwrite(".h5", a2)
    self.array_readwrite('.h5', a3)
    self.array_readwrite(".h5", a4)

    # arrayset writing tests
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for k in range(10):
      a1.append(numpy.random.normal(size=(2,3)).astype('float32'))
      a2.append(numpy.random.normal(size=(2,3,4)).astype('float64'))
      a3.append(numpy.random.normal(size=(2,3,4,5)).astype('complex128'))
      a4.append((10*numpy.random.normal(size=(3,3))).astype('uint64'))

    self.arrayset_readwrite('.h5', a1)
    self.arrayset_readwrite(".h5", a2)
    self.arrayset_readwrite('.h5', a3)
    self.arrayset_readwrite(".h5", a4)

    # complete transcoding tests
    self.transcode('test1.hdf5')
    self.transcode('matlab_1d.hdf5')
    self.transcode('matlab_2d.hdf5')

  def test01_t3binary(self):

    # array writing tests
    a1 = numpy.random.normal(size=(3,4)).astype('float32') #good, supported
    a2 = numpy.random.normal(size=(3,4)).astype('float64') #good, supported
    a3 = numpy.random.normal(size=(3,4)).astype('complex128') #not supported

    self.array_readwrite('.bindata', a1)
    self.array_readwrite(".bindata", a2)
    self.assertRaises(TypeError, self.array_readwrite, ".bindata", a3)

    # arrayset writing tests
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for k in range(10):
      a1.append(numpy.random.normal(size=(24,)).astype('float32')) #supported
      a2.append(numpy.random.normal(size=(24,)).astype('float64')) #supported
      a3.append(numpy.random.normal(size=(24,)).astype('complex128')) #unsupp.
      a4.append(numpy.random.normal(size=(3,3))) #not supported

    self.arrayset_readwrite('.bindata', a1)
    self.arrayset_readwrite(".bindata", a2)
    self.assertRaises(TypeError, self.arrayset_readwrite, ".bindata", a3)
    self.assertRaises(RuntimeError, self.arrayset_readwrite, ".bindata", a4)

    # complete transcoding test
    self.transcode('torch3.bindata')

  def test02_matlab(self):

    # array writing tests
    a1 = numpy.random.normal(size=(2,3)).astype('float32')
    a2 = numpy.random.normal(size=(2,3,4)).astype('float64')
    a3 = numpy.random.normal(size=(2,3,4,5)).astype('complex128')
    a4 = (10 * numpy.random.normal(size=(3,3))).astype('uint64')

    self.array_readwrite('.mat', a1)
    self.array_readwrite(".mat", a2)
    self.array_readwrite('.mat', a3)
    self.array_readwrite(".mat", a4)

    # arrayset writing tests
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for k in range(10):
      a1.append(numpy.random.normal(size=(2,3)).astype('float32'))
      a2.append(numpy.random.normal(size=(2,3,4)).astype('float64'))
      a3.append(numpy.random.normal(size=(2,3,4,5)).astype('complex128'))
      a4.append((10*numpy.random.normal(size=(3,3))).astype('uint64'))

    self.arrayset_readwrite('.mat', a1)
    self.arrayset_readwrite(".mat", a2)
    self.arrayset_readwrite('.mat', a3)
    self.arrayset_readwrite(".mat", a4)

    # complete transcoding tests
    self.transcode('test_1d.mat') #pseudo 1D - matlab does not support true 1D
    self.transcode('test_2d.mat')
    self.transcode('test_3d.mat')
    self.transcode('test_4d.mat')
    self.transcode('test_1d_cplx.mat') #pseudo 1D - matlab does not support 1D
    self.transcode('test_2d_cplx.mat')
    self.transcode('test_3d_cplx.mat')
    self.transcode('test_4d_cplx.mat')
    self.transcode('test.mat') #3D complex, large

  def test03_tensorfile(self):
    
    # array writing tests
    a1 = numpy.random.normal(size=(3,4)).astype('float32')
    a2 = numpy.random.normal(size=(3,4,5)).astype('float64')
    a3 = (100*numpy.random.normal(size=(2,3,4,5))).astype('int32')

    self.array_readwrite('.tensor', a1)
    self.array_readwrite(".tensor", a2)
    self.array_readwrite(".tensor", a3)

    # arrayset writing tests
    a1 = []
    a2 = []
    a3 = []
    for k in range(10):
      a1.append(numpy.random.normal(size=(3,4)).astype('float32'))
      a2.append(numpy.random.normal(size=(3,4,5)).astype('float64'))
      a3.append((100*numpy.random.normal(size=(2,3,4,5))).astype('int32'))

    self.arrayset_readwrite('.tensor', a1)
    self.arrayset_readwrite(".tensor", a2)
    self.arrayset_readwrite(".tensor", a3)

    # complete transcoding test
    self.transcode('torch.tensor')

  def test04_image(self):

    def image_transcode(filename):
        
      tmpname = tempname(os.path.splitext(filename)[1])

      try:
        # complete transcoding test
        image = torch.io.open(filename, 'r').read()

        # save with the same extension
        outfile = torch.io.open(filename, 'w')
        outfile.write(image)

        # reload the image from the file
        image2 = torch.io.open(filename, 'r').read()

        self.assertTrue ( numpy.array_equal(image, image2) )

      finally:
        if os.path.exists(tmpname): os.unlink(tmpname)

    image_transcode('test.pgm') #indexed, works fine
    image_transcode('test.pbm') #indexed, works fine
    image_transcode('test.ppm') #indexed, works fine
    #image_transcode('test.jpg') #does not work because of re-compression

  def xtest05_bin(self):

    # DEPRECATED

    # array writing tests
    a1 = numpy.random.normal(size=(2,3)).astype('float32')
    a2 = numpy.random.normal(size=(2,3,4)).astype('float64')
    a3 = numpy.random.normal(size=(2,3,4,5)).astype('complex128')
    a4 = (10 * numpy.random.normal(size=(3,3))).astype('uint64')

    self.array_readwrite('.bin', a1)
    self.array_readwrite(".bin", a2)
    self.array_readwrite('.bin', a3)
    self.array_readwrite(".bin", a4)

    # arrayset writing tests
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for k in range(10):
      a1.append(numpy.random.normal(size=(2,3)).astype('float32'))
      a2.append(numpy.random.normal(size=(2,3,4)).astype('float64'))
      a3.append(numpy.random.normal(size=(2,3,4,5)).astype('complex128'))
      a4.append((10*numpy.random.normal(size=(3,3))).astype('uint64'))

    self.arrayset_readwrite('.bin', a1)
    self.arrayset_readwrite(".bin", a2)
    self.arrayset_readwrite('.bin', a3)
    self.arrayset_readwrite(".bin", a4)

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
