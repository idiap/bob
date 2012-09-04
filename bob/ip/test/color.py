#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri Mar 25 13:44:43 2011 +0100
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

"""Test color conversions available in bob
"""

import os, sys
import platform
import numpy
import unittest
import bob
import colorsys

class ColorTest(unittest.TestCase):
  """Various color conversion tests."""

  def test01_hsv(self):

    # This test verifies that color conversion is reversible for HSV <=> RGB
    # It also shows how you can convert single pixel representations from RGB
    # to HSV and vice-versa, testing the output of bob color converters
    # against python's.

    step = 0.02
    for r in numpy.arange(0, 1, step):
      for g in numpy.arange(0, 1, step):
        for b in numpy.arange(0, 1, step):
          # First test the correctness
          ht, st, vt = bob.ip.rgb_to_hsv_f(r, g, b)
          hp, sp, vp = colorsys.rgb_to_hsv(r, g, b)
          self.assertTrue(abs(ht - hp) < 1e-6)
          self.assertTrue(abs(st - sp) < 1e-6)
          self.assertTrue(abs(vt - vp) < 1e-6)
          # And that we can invert the result using bob
          r2, g2, b2 = bob.ip.hsv_to_rgb_f(ht, st, vt)
          self.assertTrue(abs(r2 - r) < 1e-6)
          self.assertTrue(abs(g2 - g) < 1e-6)
          self.assertTrue(abs(b2 - b) < 1e-6)

  def test02_hsl(self):

    # This test verifies that color conversion is reversible for HSL <=> RGB
    # It also shows how you can convert single pixel representations from RGB
    # to HSL and vice-versa, testing the output of bob color converters
    # against python's.

    step = 0.02
    for r in numpy.arange(0, 1, step):
      for g in numpy.arange(0, 1, step):
        for b in numpy.arange(0, 1, step):
          # First test the correctness
          ht, st, lt = bob.ip.rgb_to_hsl_f(r, g, b)
          hp, lp, sp = colorsys.rgb_to_hls(r, g, b)
          self.assertTrue(abs(ht - hp) < 1e-6)
          self.assertTrue(abs(st - sp) < 1e-6)
          self.assertTrue(abs(lt - lp) < 1e-6)
          # And that we can invert the result using bob
          r2, g2, b2 = bob.ip.hsl_to_rgb_f(ht, st, lt)
          self.assertTrue(abs(r2 - r) < 1e-6)
          self.assertTrue(abs(g2 - g) < 1e-6)
          self.assertTrue(abs(b2 - b) < 1e-6)

  def test03_yuv(self):

    # This test verifies that color conversion is reversible for YUV <=> RGB
    # It also shows how you can convert single pixel representations from RGB
    # to YUV and vice-versa.

    step = 0.02
    for r in numpy.arange(0, 1, step):
      for g in numpy.arange(0, 1, step):
        for b in numpy.arange(0, 1, step):
          # First test the correctness
          yt, ut, vt = bob.ip.rgb_to_yuv_f(r, g, b)
          # And that we can invert the result using bob
          r2, g2, b2 = bob.ip.yuv_to_rgb_f(yt, ut, vt)
          self.assertTrue(abs(r2 - r) < 1e-4)
          self.assertTrue(abs(g2 - g) < 1e-4)
          self.assertTrue(abs(b2 - b) < 1e-4)

  def test04_int_conversions(self):

    # You can also use integer based conversions in which case the ranges
    # should occupy the whole valid range for the type. We support unsigned
    # integers with 8 (uint8_t) or 16 bits (uint16_t). The ranges are 0 to 255
    # for 8-bit unsigned integers and 0 to 65535 for 16-bit unsigned ones. '0'
    # represents total black while the maximum value, total white. Internally,
    # bob converts the integers into float representations and calculate the
    # the conversions just like in tests 01 to 03 above. The last step is a
    # back conversion into integer scale. This procedure may lead differences
    # in the representations and the two-way conversion. 

    # Just test a subrange or the test will take too long

    # Expected errors
    #
    # OSX 10.6 |    HSV     |    HSL     |    YUV  
    # ---------+------------+------------+-------------
    # uint8_t  | (3) 1.18%  | (4) 1.57%  | (1) 0.39%  
    # uint16_t | (3) 0.005% | (4) 0.006% | (1) 0.006%

    mx = 4
    for r in range(0,5) + range(120,130) + range(253,256):
      for g in range(0,6) + range(125,135) + range(252,256):
        for b in range(0,7) + range(127,137) + range(252,256):
          ht, st, vt = bob.ip.rgb_to_hsv_u8(r, g, b)
          r2, g2, b2 = bob.ip.hsv_to_rgb_u8(ht, st, vt)
          #mx2 = max(abs(r2-r), abs(g2-g), abs(b2-b))
          #correct within a 2% margin 
          #if mx2 > mx and (mx2/255.) < 0.02: mx = mx2
          self.assertTrue(abs(r2 - r) <= mx)
          self.assertTrue(abs(g2 - g) <= mx)
          self.assertTrue(abs(b2 - b) <= mx)
    #print "uint8_t RGB/HSV/RGB error: %d (%.2f%%)" % (mx, 100*mx/255.) 

    mx = 5
    for r in range(0,5) + range(120,130) + range(253,256):
      for g in range(0,6) + range(125,135) + range(252,256):
        for b in range(0,7) + range(127,137) + range(252,256):
          ht, st, lt = bob.ip.rgb_to_hsl_u8(r, g, b)
          r2, g2, b2 = bob.ip.hsl_to_rgb_u8(ht, st, lt)
          #mx2 = max(abs(r2-r), abs(g2-g), abs(b2-b))
          #correct within a 2% margin 
          #if mx2 > mx and (mx2/255.) < 0.02: mx = mx2
          self.assertTrue(abs(r2 - r) <= mx)
          self.assertTrue(abs(g2 - g) <= mx)
          self.assertTrue(abs(b2 - b) <= mx)
    #print "uint8_t RGB/HSL/RGB error: %d (%.2f%%)" % (mx, 100*mx/255.) 

    mx = 2
    for r in range(0,5) + range(120,130) + range(253,256):
      for g in range(0,6) + range(125,135) + range(252,256):
        for b in range(0,7) + range(127,137) + range(252,256):
          yt, ut, vt = bob.ip.rgb_to_yuv_u8(r, g, b)
          r2, g2, b2 = bob.ip.yuv_to_rgb_u8(yt, ut, vt)
          #mx2 = max(abs(r2-r), abs(g2-g), abs(b2-b))
          #correct within a 2% margin 
          #if mx2 > mx and (mx2/255.) < 0.02: mx = mx2
          self.assertTrue(abs(r2 - r) <= mx)
          self.assertTrue(abs(g2 - g) <= mx)
          self.assertTrue(abs(b2 - b) <= mx)
    #print "uint8_t RGB/YCbCr/RGB error: %d (%.2f%%)" % (mx, 100*mx/255.)

    # Just test a subrange or the test will take too long
    mx = 3
    for r in range(0,5) + range(30000,30005) + range(65530,65536):
      for g in range(0,6) + range(30002,3007) + range(65525,65532):
        for b in range(0,7) + range(3003,3008) + range(65524,65531):
          ht, st, vt = bob.ip.rgb_to_hsv_u16(r, g, b)
          r2, g2, b2 = bob.ip.hsv_to_rgb_u16(ht, st, vt)
          #mx2 = max(abs(r2-r), abs(g2-g), abs(b2-b))
          #if mx2 > mx and (mx2/65535.) < 0.0001: mx = mx2
          self.assertTrue(abs(r2 - r) <= mx)
          self.assertTrue(abs(g2 - g) <= mx)
          self.assertTrue(abs(b2 - b) <= mx)
    #print "16-bit unsigned integer RGB/HSV/RGB error: %d (%.4f%%)" % (mx, 100*mx/65535.) 

    mx = 4
    for r in range(0,5) + range(30000,30005) + range(65530,65536):
      for g in range(0,6) + range(30002,3007) + range(65525,65532):
        for b in range(0,7) + range(3003,3008) + range(65524,65531):
          ht, st, lt = bob.ip.rgb_to_hsl_u16(r, g, b)
          r2, g2, b2 = bob.ip.hsl_to_rgb_u16(ht, st, lt)
          #mx2 = max(abs(r2-r), abs(g2-g), abs(b2-b))
          #if mx2 > mx and (mx2/65535.) < 0.0001: mx = mx2
          self.assertTrue(abs(r2 - r) <= mx)
          self.assertTrue(abs(g2 - g) <= mx)
          self.assertTrue(abs(b2 - b) <= mx)
    #print "16-bit unsigned integer RGB/HSL/RGB error: %d (%.4f%%)" % (mx, 100*mx/65535.) 

    mx = 4
    for r in range(0,10) + range(120,130) + range(250,256):
      for g in range(5,12) + range(125,135) + range(240,252):
        for b in range(7,15) + range(127,137) + range(235,251):
          yt, ut, vt = bob.ip.rgb_to_yuv_u16(r, g, b)
          r2, g2, b2 = bob.ip.yuv_to_rgb_u16(yt, ut, vt)
          #mx2 = max(abs(r2-r), abs(g2-g), abs(b2-b))
          #if mx2 > mx and (mx2/65535.) < 0.0001: mx = mx2
          self.assertTrue(abs(r2 - r) <= mx)
          self.assertTrue(abs(g2 - g) <= mx)
          self.assertTrue(abs(b2 - b) <= mx)
    #print "16-bit unsigned integer RGB/YCbCr/RGB error: %d (%.4f%%)" % (mx, 100*mx/65535.) 

  def test05_gray_halves(self):

    # tests some border-line cases for gray-scale conversion and makes sure
    # we don't have precision problems.

    if platform.architecture()[0] == '64bit': 
      # do a full test, require all values to be the same
      correct = bob.io.load('gray-u8-mids.hdf5')
      for k in range(correct.shape[0]):
        self.assertEqual(correct[k,3], 
            bob.ip.rgb_to_gray_u8(*[int(z) for z in correct[k,:3]]) 
            )
    else:
      # do a full test, require all values to be the same
      # use a special 32-bit file. About 1600 cases do not match the 64-bit
      # ones.
      correct = bob.io.load('gray-u8-mids-32bits.hdf5')
      for k in range(correct.shape[0]):
        self.assertEqual(correct[k,3], 
            bob.ip.rgb_to_gray_u8(*[int(z) for z in correct[k,:3]]) 
            )

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(ColorTest)
