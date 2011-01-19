from libpytorch_ip import *

def spixrgb_str(v):
  return 'RGB(%d, %d, %d)' % (v.r, v.g, v.b)
sPixRGB.__str__ = spixrgb_str

def spixyuv_str(v):
  return 'YUV(%d, %d, %d)' % (v.y, v.u, v.v)
sPixYUV.__str__ = spixyuv_str

def spoint2d_str(v):
  return 'Point2D(x=%.3e, y=%.3e)' % (v.x, v.y)
sPoint2D.__str__ = spoint2d_str

def spoint2dpolar_str(v):
  return 'Point2Dpolar(rho=%.3e, theta=%.3e)' % (v.rho, v.theta)
sPoint2Dpolar.__str__ = spoint2dpolar_str

def srect2d_str(v):
  return 'Rect2D(x=%d, y=%d, width=%d, height=%d)' % (v.x, v.y, v.w, v.h)
sRect2D.__str__ = srect2d_str

def ssize_str(v):
  return 'Size(width=%d, height=%d)' % (v.w, v.h)
sSize.__str__ = ssize_str

def srect2dpolar_str(v):
  return 'Rect2Dpolar(tl=' + str(v.tl) + ', ' + 'tr=' + str(v.tr) + ', ' + \
      'bl=' + str(v.bl) + ', ' + 'br=' + str(v.br) + ')'
sRect2Dpolar.__str__ = srect2dpolar_str

def scomplex_str(v):
  return 'Complex(r=%.3e, i=%.3e)' % (v.r, v.i)
sComplex.__str__ = scomplex_str

def color_str(c):
  return 'Color(%s, %d, %d, %d)' % (c.coding, c.data0, c.data1, c.data2)
Color.__str__ = color_str
Color.__repr__ = color_str

def color_eq(c1, c2):
  return ((c1.coding == c2.coding) and (c1.data0 == c2.data0) and \
      (c1.data1 == c2.data1) and (c1.data2 == c2.data2))
Color.__eq__ = color_eq

def image_as_gray(i):
  if i.getNPlanes() == 1: return i
  return i._toGray()
Image.as_gray = image_as_gray

def image_as_rgb(i):
  if i.getNPlanes() == 3: return i
  return i._toRGB()
Image.as_rgb = image_as_rgb

def image_str(i):
  return 'Image(width=%d, height=%d, planes=%d)' % (i.width, i.height, i.nplanes)
Image.__str__ = image_str

__all__ = dir()
