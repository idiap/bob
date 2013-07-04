#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Nov 16 13:27:15 2011 +0100
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

import os
import sys
import numpy
import nose.tools

from .. import load, write, File
from ...test import utils as testutils

def transcode(filename):
  """Runs a complete transcoding test, to and from the binary format."""

  tmpname = testutils.temporary_filename(suffix=os.path.splitext(filename)[1])

  try:
    # transcode from test format into the test format -- test array access modes
    orig_data = load(filename)
    write(orig_data, tmpname)
    rewritten_data = load(tmpname)

    assert numpy.array_equal(orig_data, rewritten_data)

    # transcode to test format -- test arrayset access modes
    trans_file = File(tmpname, 'w')
    index = [slice(orig_data.shape[k]) for k in range(len(orig_data.shape))]
    for k in range(orig_data.shape[0]):
      index[0] = k
      trans_file.append(orig_data[index]) #slice from first dimension
    del trans_file

    rewritten_file = File(tmpname, 'r')

    for k in range(orig_data.shape[0]):
      rewritten_data = rewritten_file.read(k)
      index[0] = k
      assert numpy.array_equal(orig_data[index], rewritten_data)

  finally:
    # And we erase both files after this
    if os.path.exists(tmpname): os.unlink(tmpname)

def array_readwrite(extension, arr, close=False):
  """Runs a read/write verify step using the given numpy data"""
  tmpname = testutils.temporary_filename(suffix=extension)
  try:
    write(arr, tmpname)
    reloaded = load(tmpname)
    if close: assert numpy.allclose(arr, reloaded)
    else: assert numpy.array_equal(arr, reloaded)
  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)

def arrayset_readwrite(extension, arrays, close=False):
  """Runs a read/write verify step using the given numpy data"""
  tmpname = testutils.temporary_filename(suffix=extension)
  try:
    f = File(tmpname, 'w')
    for k in arrays:
      f.append(k)
    del f
    f = File(tmpname, 'r')
    for k, array in enumerate(arrays):
      reloaded = f.read(k) #read the contents
      if close:
        assert numpy.allclose(array, reloaded)
      else: assert numpy.array_equal(array, reloaded)
  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)

def test_hdf5():

  # array writing tests
  a1 = numpy.random.normal(size=(2,3)).astype('float32')
  a2 = numpy.random.normal(size=(2,3,4)).astype('float64')
  a3 = numpy.random.normal(size=(2,3,4,5)).astype('complex128')
  a4 = (10 * numpy.random.normal(size=(3,3))).astype('uint64')

  array_readwrite('.hdf5', a1) # extensions: .hdf5 or .h5
  array_readwrite(".h5", a2)
  array_readwrite('.h5', a3)
  array_readwrite(".h5", a4)

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

  arrayset_readwrite('.h5', a1)
  arrayset_readwrite(".h5", a2)
  arrayset_readwrite('.h5', a3)
  arrayset_readwrite(".h5", a4)

  # complete transcoding tests
  transcode(testutils.datafile('test1.hdf5', __name__))
  transcode(testutils.datafile('matlab_1d.hdf5', __name__))
  transcode(testutils.datafile('matlab_2d.hdf5', __name__))

@testutils.extension_available('.bindata')
def test_torch3_binary():

  # array writing tests
  a1 = numpy.random.normal(size=(3,4)).astype('float32') #good, supported
  a2 = numpy.random.normal(size=(3,4)).astype('float64') #good, supported
  a3 = numpy.random.normal(size=(3,4)).astype('complex128') #not supported

  array_readwrite('.bindata', a1)
  array_readwrite(".bindata", a2)
  nose.tools.assert_raises(TypeError, array_readwrite, ".bindata", a3)

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

  arrayset_readwrite('.bindata', a1)
  arrayset_readwrite(".bindata", a2)

  # checks we raise if we don't suppport a type
  nose.tools.assert_raises(TypeError, arrayset_readwrite, ".bindata", a3)
  nose.tools.assert_raises(RuntimeError, arrayset_readwrite, ".bindata", a4)

  # complete transcoding test
  transcode(testutils.datafile('torch3.bindata', __name__))

@testutils.extension_available('.mat')
def test_mat_file_io():

  # array writing tests
  a1 = numpy.random.normal(size=(2,3)).astype('float32')
  a2 = numpy.random.normal(size=(2,3,4)).astype('float64')
  a3 = numpy.random.normal(size=(2,3,4,5)).astype('complex128')
  a4 = (10 * numpy.random.normal(size=(3,3))).astype('uint64')

  array_readwrite('.mat', a1)
  array_readwrite(".mat", a2)
  array_readwrite('.mat', a3)
  array_readwrite(".mat", a4)

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

  arrayset_readwrite('.mat', a1)
  arrayset_readwrite(".mat", a2)
  arrayset_readwrite('.mat', a3)
  arrayset_readwrite(".mat", a4)

  # complete transcoding tests
  transcode(testutils.datafile('test_1d.mat', __name__)) #pseudo 1D - matlab does not support true 1D
  transcode(testutils.datafile('test_2d.mat', __name__))
  transcode(testutils.datafile('test_3d.mat', __name__))
  transcode(testutils.datafile('test_4d.mat', __name__))
  transcode(testutils.datafile('test_1d_cplx.mat', __name__)) #pseudo 1D - matlab does not support 1D
  transcode(testutils.datafile('test_2d_cplx.mat', __name__))
  transcode(testutils.datafile('test_3d_cplx.mat', __name__))
  transcode(testutils.datafile('test_4d_cplx.mat', __name__))
  transcode(testutils.datafile('test.mat', __name__)) #3D complex, large

@nose.tools.nottest
@testutils.extension_available('.mat')
def test_mat_file_io_does_not_crash():

  data = load(testutils.datafile('test_cell.mat', __name__))

@testutils.extension_available('.tensor')
def test_tensorfile():

  # array writing tests
  a1 = numpy.random.normal(size=(3,4)).astype('float32')
  a2 = numpy.random.normal(size=(3,4,5)).astype('float64')
  a3 = (100*numpy.random.normal(size=(2,3,4,5))).astype('int32')

  array_readwrite('.tensor', a1)
  array_readwrite(".tensor", a2)
  array_readwrite(".tensor", a3)

  # arrayset writing tests
  a1 = []
  a2 = []
  a3 = []
  for k in range(10):
    a1.append(numpy.random.normal(size=(3,4)).astype('float32'))
    a2.append(numpy.random.normal(size=(3,4,5)).astype('float64'))
    a3.append((100*numpy.random.normal(size=(2,3,4,5))).astype('int32'))

  arrayset_readwrite('.tensor', a1)
  arrayset_readwrite(".tensor", a2)
  arrayset_readwrite(".tensor", a3)

  # complete transcoding test
  transcode(testutils.datafile('torch.tensor', __name__))

@testutils.extension_available('.pgm')
@testutils.extension_available('.pbm')
@testutils.extension_available('.ppm')
def test_netpbm():

  def image_transcode(filename):

    tmpname = testutils.temporary_filename(suffix=os.path.splitext(filename)[1])

    try:
      # complete transcoding test
      image = load(filename)

      # save with the same extension
      write(image, tmpname)

      # reload the image from the file
      image2 = load(tmpname)

      assert numpy.array_equal(image, image2)

    finally:
      if os.path.exists(tmpname): os.unlink(tmpname)

  image_transcode(testutils.datafile('test.pgm', __name__)) #indexed, works fine
  image_transcode(testutils.datafile('test.pbm', __name__)) #indexed, works fine
  image_transcode(testutils.datafile('test.ppm', __name__)) #indexed, works fine
  #image_transcode(testutils.datafile('test.jpg', __name__)) #does not work because of re-compression

@testutils.extension_available('.csv')
def test_csv():

  # array writing tests
  a1 = numpy.random.normal(size=(5,5)).astype('float64')
  a2 = numpy.random.normal(size=(5,10)).astype('float64')
  a3 = numpy.random.normal(size=(5,100)).astype('float64')

  array_readwrite('.csv', a1, close=True)
  array_readwrite(".csv", a2, close=True)
  array_readwrite('.csv', a3, close=True)

  # arrayset writing tests
  a1 = []
  a2 = []
  a3 = []
  for k in range(10):
    a1.append(numpy.random.normal(size=(5,)).astype('float64'))
    a2.append(numpy.random.normal(size=(50,)).astype('float64'))
    a3.append(numpy.random.normal(size=(500,)).astype('float64'))

  arrayset_readwrite('.csv', a1, close=True)
  arrayset_readwrite(".csv", a2, close=True)
  arrayset_readwrite('.csv', a3, close=True)
