#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 25 Jul 13:03:23 2011 

"""Tests some drawing on images
"""

import os, sys
import unittest
import torch
import numpy

class DrawingTest(unittest.TestCase):
  """Various drawing tests."""

  def test01_gray_point(self):

    # Tests single point drawing using gray-scaled images
    image = numpy.ndarray((100, 100), 'uint8')
    image.fill(0)

    # Draws a white point on the middle
    torch.ip.draw_point(image, 50, 50, 255)
    self.assertEqual(image[50, 50], 255)

    # Try drawing on an unexisting location, should not raise
    imcopy = image.copy()
    torch.ip.try_draw_point(imcopy, 100, 100, 255)
    self.assertTrue( numpy.array_equal(image, imcopy) ) # no change is made

    # Try drawing with draw_point on an unexisting location, should raise
    self.assertRaises(IndexError, torch.ip.draw_point, imcopy, 100, 100, 255) 

  def test02_color_point(self):

    # color
    white = (255, 255, 255) #rgb
    a1    = numpy.ndarray((3,), 'uint8')
    a1.fill(255) #a comparision array

    # Tests single point drawing using gray-scaled images
    image = numpy.ndarray((3, 100, 100), 'uint8')
    image.fill(0)

    # Draws a white point on the middle
    torch.ip.draw_point(image, 50, 50, white)
    self.assertTrue(numpy.array_equal(image[:,50, 50],a1))

    # Try drawing on an unexisting location, should not raise
    imcopy = image.copy()
    torch.ip.try_draw_point(imcopy, 100, 100, white)
    self.assertTrue(numpy.array_equal(image, imcopy)) # no change is made

    # Try drawing with draw_point on an unexisting location, should raise
    self.assertRaises(IndexError, torch.ip.draw_point, imcopy, 100, 100, white)

  def test03_line(self):

    # draws a gray line, test to see if works; note the same algorithm is used
    # for color line plotting, so we only test the gray one.

    image = numpy.ndarray((100, 100), 'uint8')
    image.fill(0)

    # Draws a white line on the middle (horizontal)
    torch.ip.draw_line(image, 50, 50, 50, 70, 255)
    for k in range(50,70):
      self.assertEqual(image[k,50], 255)

    # Draws a white line on the middle (vertical)
    torch.ip.draw_line(image, 50, 50, 70, 50, 230)
    for k in range(50,70):
      self.assertEqual(image[50,k], 230)

    # Draws a white line on the middle (horizontal, backwards)
    torch.ip.draw_line(image, 50, 70, 50, 50, 128)
    for k in range(50,70):
      self.assertEqual(image[k,50], 128)

    # Draws a white line on the middle (vertical, backwards)
    torch.ip.draw_line(image, 70, 50, 50, 50, 65)
    for k in range(50,70):
      self.assertEqual(image[50,k], 65)

  def test04_box(self):

    # draws a box on the image, only test gray as color uses the same
    # algorithm.

    image = numpy.ndarray((100, 100), 'uint8')
    image.fill(0)

    # Draws a white line on the middle (horizontal)
    torch.ip.draw_box(image, 50, 50, 20, 20, 255)

    for k in range(50,70):
      self.assertEqual(image[k,50], 255)
      self.assertEqual(image[50,k], 255)
      self.assertEqual(image[k,70], 255)
      self.assertEqual(image[70,k], 255)

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

