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

