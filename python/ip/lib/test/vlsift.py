#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Mon Dec 19 19:12:19 2011 +0100
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

"""Tests our SIFT features extractor
"""

import os, sys
import unittest
import platform
import bob
import numpy

def equal(x, y, epsilon):
  return (abs(x - y) < epsilon)

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class VLSiftTest(unittest.TestCase):
  """Performs various tests"""

  def test01_VLSiftPython(self):
    img = bob.io.load(os.path.join("sift", "vlimg_ref.pgm"))
    if platform.architecture()[0] == '64bit':
      ref = bob.io.Arrayset(os.path.join("sift", "vlsift_ref.hdf5"))
    else:
      ref = bob.io.Arrayset(os.path.join("sift", "vlsift_ref_32bits.hdf5"))
    mysift1 = bob.ip.VLSIFT(478,640, 3, 6, -1)
    out = mysift1(img)
    self.assertTrue(len(out) == len(ref))
    for i in range(len(out)):
      # Forces the cast in VLFeat sift main() function
      outi_uint8 = numpy.array( out[i], dtype='uint8') 
      out[i][4:132] = outi_uint8[4:132]
      self.assertTrue(equals(out[i],ref[i],1e-3))

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(VLSiftTest)
