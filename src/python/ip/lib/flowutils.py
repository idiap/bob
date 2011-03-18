#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 18 Mar 13:13:18 2011 

"""A few common utilities that are useful in Optical Flow studies.
"""

from ..core.array import radius, atan2, uint8_3
import math
import colorsys

def flow2color(u, v):
  """Calculates a color-coded image that represents the Optical Flow.
  
  This is what we do:
  
  1. Convert u,v to polar coordinates
  2. Calculate the maximum radius and normalize all others by this value. This
  gives the intensity of the color. The closer to the origin, the whiter
  3. The angle derive the hue of the color, zero begin 0 radians and 1 = pi
  radians.
  """
  
  # polar coordinate conversion using blitz
  r = radius(u, v)
  t = atan2(u, v)

  # calculates hue and lightness (value is always == 1)
  hue = abs(t)/math.pi
  r /= r.max()
  lightness = -r + 1

  image = uint8_3(3, u.extent(0), u.extent(1)) # planes: red, green, blue
  for i in range(0, image.extent(1)):
    for j in range(0, image.extent(2)):
      r, g, b = colorsys.hls_to_rgb(hue[i,j], lightness[i,j], 1.0)
      image[0,i,j] = int(255*r)
      image[1,i,j] = int(255*g)
      image[2,i,j] = int(255*b)

  return image
