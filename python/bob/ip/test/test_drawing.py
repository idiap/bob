#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon Jul 25 14:02:56 2011 +0200
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

"""Tests some drawing on images
"""

import os, sys
import unittest
import bob
import numpy

class DrawingTest(unittest.TestCase):
  """Various drawing tests."""

  def test01_gray_point(self):

    # Tests single point drawing using gray-scaled images
    image = numpy.ndarray((100, 100), 'uint8')
    image.fill(0)

    # Draws a white point on the middle
    bob.ip.draw_point(image, 50, 50, 255)
    self.assertEqual(image[50, 50], 255)

    # Try drawing on an unexisting location, should not raise
    imcopy = image.copy()
    bob.ip.try_draw_point(imcopy, 100, 100, 255)
    self.assertTrue( numpy.array_equal(image, imcopy) ) # no change is made

    # Try drawing with draw_point on an unexisting location, should raise
    self.assertRaises(IndexError, bob.ip.draw_point, imcopy, 100, 100, 255) 

  def test02_color_point(self):

    # color
    white = (255, 255, 255) #rgb
    a1    = numpy.ndarray((3,), 'uint8')
    a1.fill(255) #a comparision array

    # Tests single point drawing using gray-scaled images
    image = numpy.ndarray((3, 100, 100), 'uint8')
    image.fill(0)

    # Draws a white point on the middle
    bob.ip.draw_point(image, 50, 50, white)
    self.assertTrue(numpy.array_equal(image[:,50, 50],a1))

    # Try drawing on an unexisting location, should not raise
    imcopy = image.copy()
    bob.ip.try_draw_point(imcopy, 100, 100, white)
    self.assertTrue(numpy.array_equal(image, imcopy)) # no change is made

    # Try drawing with draw_point on an unexisting location, should raise
    self.assertRaises(IndexError, bob.ip.draw_point, imcopy, 100, 100, white)

  def test03_line(self):

    # draws a gray line, test to see if works; note the same algorithm is used
    # for color line plotting, so we only test the gray one.

    image = numpy.ndarray((100, 100), 'uint8')
    image.fill(0)

    # Draws a white line on the middle (horizontal)
    bob.ip.draw_line(image, 50, 50, 50, 70, 255)
    for k in range(50,70):
      self.assertEqual(image[k,50], 255)

    # Draws a white line on the middle (vertical)
    bob.ip.draw_line(image, 50, 50, 70, 50, 230)
    for k in range(50,70):
      self.assertEqual(image[50,k], 230)

    # Draws a white line on the middle (horizontal, backwards)
    bob.ip.draw_line(image, 50, 70, 50, 50, 128)
    for k in range(50,70):
      self.assertEqual(image[k,50], 128)

    # Draws a white line on the middle (vertical, backwards)
    bob.ip.draw_line(image, 70, 50, 50, 50, 65)
    for k in range(50,70):
      self.assertEqual(image[50,k], 65)

  def test04_box(self):

    # draws a box on the image, only test gray as color uses the same
    # algorithm.

    image = numpy.ndarray((100, 100), 'uint8')
    image.fill(0)

    # Draws a white line on the middle (horizontal)
    bob.ip.draw_box(image, 50, 50, 20, 20, 255)

    for k in range(50,70):
      self.assertEqual(image[k,50], 255)
      self.assertEqual(image[50,k], 255)
      self.assertEqual(image[k,70], 255)
      self.assertEqual(image[70,k], 255)
