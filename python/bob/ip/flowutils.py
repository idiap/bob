#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 18 Mar 13:13:18 2011 

"""A few common utilities that are useful in Optical Flow studies.
"""

import math
import numpy

def flow2hsv(u, v):
  """Calculates a color-coded image that represents the Optical Flow from a
  hue-saturation-value perspective.
  
  This is what we do:
  
  1. Convert u,v to polar coordinates
  2. Calculate the maximum radius and normalize all others by this value. This
  gives the intensity of the color. The closer to the origin, the whiter
  3. The angle derive the hue of the color, zero begin 0 radians and 1 = pi
  radians.

  Parameters:
  u -- x-direction (width) velocities as floats
  v -- y-direction (height) velocities as floats

  Outputs a HSV image representation (3D, float32).
  """
  from . import hsv_to_rgb
  
  # polar coordinate conversion using blitz
  t = numpy.arctan2(v, u)
  r = numpy.sqrt(u**2 + v**2)

  # calculates hue and saturation (value is always == 1)
  hsv = numpy.ndarray((3, u.shape[0], u.shape[1]), 'float64')
  hsv[0,:,:] = abs(t)/math.pi #hue
  r /= r.max()
  hsv[1,:,:] = r #saturation
  hsv[2,:,:] = 1.0 #value

  # convert to rgb
  rgb = numpy.zeros(hsv.shape, 'float64')
  hsv_to_rgb(hsv, rgb)
  return rgb
