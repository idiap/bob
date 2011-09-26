#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 18 Mar 13:13:18 2011 

"""A few common utilities that are useful in Optical Flow studies.
"""

import math

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

  Outputs a HSV image representation (float32_3).
  """
  from ..core.array import radius, atan2, float64_3, float32_3
  from . import hsv_to_rgb
  
  # polar coordinate conversion using blitz
  r = radius(u, v)
  t = atan2(u, v)

  # calculates hue and saturation (value is always == 1)
  hsv = float64_3(3, u.extent(0), u.extent(1))
  hsv[0,:,:] = abs(t)/math.pi #hue
  r /= r.max()
  hsv[1,:,:] = r #saturation
  hsv[2,:,:] = 1.0 #value

  # convert to rgb
  rgb = float32_3(hsv.shape())
  rgb.fill(0)
  hsv_to_rgb(hsv.cast('float32'), rgb)
  return rgb
