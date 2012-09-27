#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Nov 16 13:27:15 2011 +0100
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""A combined test for all built-in types of Array/interaction in
python.
"""

import os, sys
import unittest
import tempfile
import bob
import numpy
import random
import pkg_resources
from nose.plugins.skip import SkipTest
import functools

def extension_available(extension):
  '''Decorator to check if a extension is available before enabling a test'''

  def test_wrapper(test):

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
      if bob.io.extensions().has_key(extension):
        return test(*args, **kwargs)
      else:
        raise SkipTest('Extension to handle "%s" files was not available at compile time' % extension)

    return wrapper

  return test_wrapper

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

DEFAULT_EXTENSION = '.hdf5' # define here the codec you trust

# This test implements a generalized framework for testing codecs. It
# loads files in the codec native format, convert into bob native binary
# format and back, comparing the outcomes at every stage. We believe in the
# quality of the binary codec because that is covered in other tests.

def tempname(suffix, prefix='bobtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

def _transcode(self, filename):
  """Runs a complete transcoding test, to and from the binary format."""

  tmpname = tempname(os.path.splitext(filename)[1])

  try:
    # transcode from test format into the test format -- test array access modes
    orig_data = bob.io.open(filename, 'r').read()
    bob.io.open(tmpname, 'w').write(orig_data)
    rewritten_data = bob.io.open(tmpname, 'r').read()

    self.assertTrue( numpy.array_equal(orig_data, rewritten_data) )

    # transcode to test format -- test arrayset access modes
    trans_file = bob.io.open(tmpname, 'w')
    index = [slice(orig_data.shape[k]) for k in range(len(orig_data.shape))]
    for k in range(orig_data.shape[0]):
      index[0] = k
      trans_file.append(orig_data[index]) #slice from first dimension
    del trans_file

    rewritten_file = bob.io.open(tmpname, 'r')

    for k in range(orig_data.shape[0]):
      rewritten_data = rewritten_file.read(k)
      index[0] = k
      self.assertTrue( numpy.array_equal(orig_data[index], rewritten_data) )

  finally:
    # And we erase both files after this
    if os.path.exists(tmpname): os.unlink(tmpname)

# We attach the transcoding method to the testcase class, so we can use its
# assertions
unittest.TestCase.transcode = _transcode

def _array_readwrite(self, extension, arr, close=False):
  """Runs a read/write verify step using the given numpy data"""
  tmpname = tempname(extension)
  try:
    f = bob.io.open(tmpname, 'w')
    f.write(arr)
    del f
    f = bob.io.open(tmpname, 'r')
    reloaded = f.read() #read the contents
    if close: self.assertTrue(numpy.allclose(arr, reloaded))
    else: self.assertTrue(numpy.array_equal(arr, reloaded))
  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)

# And we attach...
unittest.TestCase.array_readwrite = _array_readwrite

def _arrayset_readwrite(self, extension, arrays, close=False):
  """Runs a read/write verify step using the given numpy data"""
  tmpname = tempname(extension)
  try:
    f = bob.io.open(tmpname, 'w')
    for k in arrays: 
      f.append(k)
    del f
    f = bob.io.open(tmpname, 'r')
    for k, array in enumerate(arrays):
      reloaded = f.read(k) #read the contents
      if close: 
        self.assertTrue(numpy.allclose(array, reloaded))
      else: self.assertTrue(numpy.array_equal(array, reloaded))
  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)

# And we attach...
unittest.TestCase.arrayset_readwrite = _arrayset_readwrite

class FileTest(unittest.TestCase):
  """Performs various tests for the bob::io::*File types."""

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
    self.transcode(F('test1.hdf5'))
    self.transcode(F('matlab_1d.hdf5'))
    self.transcode(F('matlab_2d.hdf5'))

  @extension_available('.bindata')
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
    self.transcode(F('torch3.bindata'))

  @extension_available('.mat')
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
    self.transcode(F('test_1d.mat')) #pseudo 1D - matlab does not support true 1D
    self.transcode(F('test_2d.mat'))
    self.transcode(F('test_3d.mat'))
    self.transcode(F('test_4d.mat'))
    self.transcode(F('test_1d_cplx.mat')) #pseudo 1D - matlab does not support 1D
    self.transcode(F('test_2d_cplx.mat'))
    self.transcode(F('test_3d_cplx.mat'))
    self.transcode(F('test_4d_cplx.mat'))
    self.transcode(F('test.mat')) #3D complex, large

  @extension_available('.tensor')
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
    self.transcode(F('torch.tensor'))

  @extension_available('.pgm')
  @extension_available('.pbm')
  @extension_available('.ppm')
  def test04_image(self):

    def image_transcode(filename):
        
      tmpname = tempname(os.path.splitext(filename)[1])

      try:
        # complete transcoding test
        image = bob.io.open(filename, 'r').read()

        # save with the same extension
        outfile = bob.io.open(tmpname, 'w')
        outfile.write(image)

        # reload the image from the file
        image2 = bob.io.open(tmpname, 'r').read()

        self.assertTrue ( numpy.array_equal(image, image2) )

      finally:
        if os.path.exists(tmpname): os.unlink(tmpname)

    image_transcode(F('test.pgm')) #indexed, works fine
    image_transcode(F('test.pbm')) #indexed, works fine
    image_transcode(F('test.ppm')) #indexed, works fine
    #image_transcode(F('test.jpg')) #does not work because of re-compression

  @extension_available('.csv')
  def test05_csv(self):

    # array writing tests
    a1 = numpy.random.normal(size=(5,5)).astype('float64')
    a2 = numpy.random.normal(size=(5,10)).astype('float64')
    a3 = numpy.random.normal(size=(5,100)).astype('float64')

    self.array_readwrite('.csv', a1, close=True)
    self.array_readwrite(".csv", a2, close=True)
    self.array_readwrite('.csv', a3, close=True)

    # arrayset writing tests
    a1 = []
    a2 = []
    a3 = []
    for k in range(10):
      a1.append(numpy.random.normal(size=(5,)).astype('float64'))
      a2.append(numpy.random.normal(size=(50,)).astype('float64'))
      a3.append(numpy.random.normal(size=(500,)).astype('float64'))

    self.arrayset_readwrite('.csv', a1, close=True)
    self.arrayset_readwrite(".csv", a2, close=True)
    self.arrayset_readwrite('.csv', a3, close=True)

  @extension_available('.bin')
  def test06_bin(self):
    
    raise SkipTest, "The extension '.bin' is deprecated"

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
