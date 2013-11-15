#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tue Apr 26 17:25:41 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the LBP framework. Find attached to this test a table with expected
LBP codes.
"""

import os, sys
import bob
import math
import numpy
import nose.tools


def generate_3x3_image(image, values):
  """Generates a 3x3 image from a 9-position value vector using the following
  technique:

           +-+-+-+
           |1|2|3|
           +-+-+-+
           |8|0|4|
           +-+-+-+
           |7|6|5|
           +-+-+-+

  """
  image[1, 1] = int(values[0])
  image[0, 0] = int(values[1])
  image[0, 1] = int(values[2])
  image[0, 2] = int(values[3])
  image[1, 2] = int(values[4])
  image[2, 2] = int(values[5])
  image[2, 1] = int(values[6])
  image[2, 0] = int(values[7])
  image[1, 0] = int(values[8])



def generate_5x5_image(image, values):
  """Generates a 5x5 image from a 25-position value vector row-wise
  """
  for i in range(0,5):
    for j in range(0,5):
      image[i,j] = int(values[i*5+j])


def generate_NxMxM_image(image, values):
  """ Generates a NxMxM image from an 1x3 array with 9-position value using the following technique for M=3:

      frame (0,:,:) =
           +-+-+-+
           |1|2|3|
           +-+-+-+
           |8|0|4|
           +-+-+-+
           |7|6|5|
           +-+-+-+

     frame (1,:,:) =
           +-+-+-+
           |1|2|3|
           +-+-+-+
           |8|0|4|
           +-+-+-+
           |7|6|5|
           +-+-+-+
     .
     .
     .
     .
     .

     frame (N,:,:) =
           +-+-+-+
           |1|2|3|
           +-+-+-+
           |8|0|4|
           +-+-+-+
           |7|6|5|
           +-+-+-+

  """

  N = image.shape[0]
  M = image.shape[1]

  for i in range(N):
    #3x3
    if(M==3):
      generate_3x3_image(image[i,:,:], values[i])
    #5x5
    else: #only two options
      generate_5x5_image(image[i,:,:],values[i])



def rotate(v, size=8):
  """Rotates the LSB bit in v, making it a HSB"""
  lsb = v & 1
  v >>= 1
  return (lsb << (size-1)) | v

def calculate_lbp8r_rotinvariant_value(v):
  """Calculates the rotation invariant LBP code for a certain 8-bit value"""
  smallest = 0xff
  tmp = v
  for k in range(8):
    tmp = rotate(tmp)
    if tmp < smallest: smallest = tmp
  return smallest

def calculate_lbp8r_rotinvariant_table():
  retval = []
  map = {} #map into bob values
  last = 0
  for k in range(256):
    v = calculate_lbp8r_rotinvariant_value(k)
    if not v in map.keys():
      map[v] = last
      last += 1
    retval.append(map[v])
  return retval

def uniformity(v):
  "Says the degree of uniformity of a certain pattern"
  retval = 0
  notation = ('%8s' % bin(v)).replace(' ', '0')
  tmp = list(notation)
  current = None
  for k in tmp:
    if current is None: #initialize
      if tmp[0] != tmp[-1]: retval += 1
    else:
      if current != k: retval += 1
    current = k
  return retval

def calculate_lbp8r_u2_table():
  retval = []
  map = {} #map into bob values
  last = 1
  for k in range(256):
    if uniformity(k) <= 2:
      retval.append(k)
    else: retval.append(0)
  return retval

def calculate_lbp8r_riu2_table():
  retval = []
  for k in calculate_lbp8r_u2_table():
    retval.append(calculate_lbp8r_rotinvariant_value(k))
  return retval

def bilinear_interpolation(image, x, y):
  """Calculates the bilinear interpolation value for a certain point in the
  given image."""
  xl = int(math.floor(x))
  xh = int(math.ceil(x))
  yl = int(math.floor(y))
  yh = int(math.floor(y))
  y1 = (xh - x) * image[yl, xl] + (x - xl) * image[yl, xh]
  y2 = (xh - x) * image[yh, xl] + (x - xl) * image[yh, xh]
  retval = ((yh - y) * y1) + ((y - yl) * y2)

"""
" Helper that generate LBP-bins to check in the unit test cases
"""
class Processor:

  """
  " Helper that generate LBP-bins to check in the unit test cases
  "
  " @param operator LBP-Operator
  " @param generator Image generator function
  " @param center Coordinates of a specific operator in order to check
  " @param img_size Images sizes (square image)
  """
  def __init__(self, operator, generator, center, img_size):
    self.operator = operator
    self.generator = generator
    self.x = center[0]
    self.y = center[1]
    self.image = numpy.ndarray((img_size, img_size), 'uint8')

  def __call__(self, value):
    image = self.generator(self.image, value)
    return self.operator(self.image, self.y, self.x)



"""
" Helper that generate LBPTop-bins to check in the unit test cases
"""
class ProcessorLBPTop:

  """
  " Helper that generate LBPTop-bins to check in the unit test cases
  "
  " @param operator LBP-Operator with defined XY,XT,YT planes
  " @param generator Image generator function
  " @param img_size Images sizes (square image)
  " @param n_frames Number of Frames
  """
  def __init__(self, operator, generator, img_size, n_frames):
    self.operator = operator
    self.generator = generator

    xy_width  = img_size-(operator.xy.radius*2) #Square image
    xy_height = xy_width

    if(operator.xt.radius>operator.yt.radius):
      maxT_radius = operator.xt.radius
    else:
      maxT_radius = operator.yt.radius
    tLength   = n_frames-(maxT_radius*2)

    self.XY = numpy.empty(shape=(tLength,xy_width,xy_height),dtype='uint16')
    self.XT = numpy.empty(shape=(tLength,xy_width,xy_height),dtype='uint16')
    self.YT = numpy.empty(shape=(tLength,xy_width,xy_height),dtype='uint16')

    self.image = numpy.ndarray((n_frames,img_size, img_size), 'uint8')


  """
  "
  "  @param plane_index Index of the plane (0- XY Plane, 1- XT Plane, 2- YT Plane)
  "  @param center Coordinates of a specific operator in order to check
  """
  def __call__(self, value='',plane_index=0,operator_coordinates=(0,0,0)):

    image = self.generator(self.image, value)
    self.operator(self.image, self.XY,self.XT,self.YT)

    x = operator_coordinates[0]
    y = operator_coordinates[1]
    z = operator_coordinates[2]

    if(plane_index==0):
      return self.XY[x,y,z]
    elif(plane_index==1):
      return self.XT[x,y,z]
    else:
      return self.YT[x,y,z]




def bin(s, m=1):
  """Converts the number s into its binary representation (as a string)"""
  return str(m*s) if s<=1 else bin(s>>1, m) + str(m*(s&1))


####################################
# Here the real test functions start

def test_vanilla_4p1r():
  op = bob.ip.LBP(4,1)
  proc = Processor(op, generate_3x3_image, (1,1), 3)
  nose.tools.eq_(proc('011111111'), 0xf)
  #please note that the bob implementation of LBPs is slightly different
  #then that of the original LBP paper:
  # s(x) >= 0 => LBP digit = 1
  # s(x) <  0 => LBO digit = 0
  nose.tools.eq_(proc('100000000'), 0x0)
  nose.tools.eq_(proc('102000000'), 0x8)
  nose.tools.eq_(proc('100020000'), 0x4)
  nose.tools.eq_(proc('100000200'), 0x2)
  nose.tools.eq_(proc('100000002'), 0x1)
  nose.tools.eq_(proc('102020000'), 0xc)
  nose.tools.eq_(proc('100020200'), 0x6)
  nose.tools.eq_(proc('100000202'), 0x3)
  nose.tools.eq_(proc('102000002'), 0x9)
  nose.tools.eq_(proc('102020200'), 0xe)
  nose.tools.eq_(proc('100020202'), 0x7)
  nose.tools.eq_(proc('102000202'), 0xb)
  nose.tools.eq_(proc('102020002'), 0xd)
  nose.tools.eq_(proc('100020002'), 0x5)
  nose.tools.eq_(proc('102000200'), 0xa)
  nose.tools.eq_(proc('102020202'), 0xf)
  nose.tools.eq_(op.max_label, 0x10) # this is set to 16!

def test_rotinvariant_4p1r():
  op = bob.ip.LBP(4,1,False,False,False,False,True)
  proc = Processor(op, generate_3x3_image, (1,1), 3)
  #bob's implementation start labelling the patterns from 0
  nose.tools.eq_(proc('100000000'), 0x0) #0x0
  nose.tools.eq_(proc('102000000'), 0x1) #0x1
  nose.tools.eq_(proc('100020000'), 0x1) #0x1
  nose.tools.eq_(proc('100000200'), 0x1) #0x1
  nose.tools.eq_(proc('100000002'), 0x1) #0x1
  nose.tools.eq_(proc('102020000'), 0x2) #0x3
  nose.tools.eq_(proc('100020200'), 0x2) #0x3
  nose.tools.eq_(proc('100000202'), 0x2) #0x3
  nose.tools.eq_(proc('102000002'), 0x2) #0x3
  nose.tools.eq_(proc('100020002'), 0x3) #0x5
  nose.tools.eq_(proc('102000200'), 0x3) #0x5
  nose.tools.eq_(proc('102020200'), 0x4) #0x7
  nose.tools.eq_(proc('100020202'), 0x4) #0x7
  nose.tools.eq_(proc('102000202'), 0x4) #0x7
  nose.tools.eq_(proc('102020002'), 0x4) #0x7
  nose.tools.eq_(proc('102020202'), 0x5) #0xf
  nose.tools.eq_(op.max_label, 0x6) # this is set to 6!

def test_u2_4p1r():
  op = bob.ip.LBP(4,1,False,False,False,True)
  proc = Processor(op, generate_3x3_image, (1,1), 3)
  nose.tools.eq_(proc('100000000'), 0x1) #0x0
  nose.tools.eq_(proc('102000000'), 0x2) #0x8
  nose.tools.eq_(proc('100020000'), 0x3) #0x4
  nose.tools.eq_(proc('100000200'), 0x4) #0x2
  nose.tools.eq_(proc('100000002'), 0x5) #0x1
  nose.tools.eq_(proc('102020000'), 0x6) #0xc
  nose.tools.eq_(proc('100020200'), 0x7) #0x6
  nose.tools.eq_(proc('100000202'), 0x8) #0x3
  nose.tools.eq_(proc('102000002'), 0x9) #0x9 (by chance!)
  nose.tools.eq_(proc('102020200'), 0xa) #0xe
  nose.tools.eq_(proc('100020202'), 0xb) #0x7
  nose.tools.eq_(proc('102000202'), 0xc) #0xb
  nose.tools.eq_(proc('102020002'), 0xd) #0xd (by chance!)
  nose.tools.eq_(proc('100020002'), 0x0) #non-uniform(2) => 0x0
  nose.tools.eq_(proc('102000200'), 0x0) #non-uniform(2) => 0x0
  nose.tools.eq_(proc('102020202'), 0xe) #0xf
  nose.tools.eq_(op.max_label, 0xf) # this is set to 15!

def test_rotinvariant_u2_4p1r():
  op = bob.ip.LBP(4,1,False,False,False,True,True)
  proc = Processor(op, generate_3x3_image, (1,1), 3)
  nose.tools.eq_(proc('100000000'), 0x1) #0x0
  nose.tools.eq_(proc('102000000'), 0x2) #0x8
  nose.tools.eq_(proc('100020000'), 0x2) #0x4
  nose.tools.eq_(proc('100000200'), 0x2) #0x2
  nose.tools.eq_(proc('100000002'), 0x2) #0x1
  nose.tools.eq_(proc('102020000'), 0x3) #0xc
  nose.tools.eq_(proc('100020200'), 0x3) #0x6
  nose.tools.eq_(proc('100000202'), 0x3) #0x3
  nose.tools.eq_(proc('102000002'), 0x3) #0x9 #missing from bob
  nose.tools.eq_(proc('102020200'), 0x4) #0xe
  nose.tools.eq_(proc('100020202'), 0x4) #0x7
  nose.tools.eq_(proc('102000202'), 0x4) #0xb #missing from bob
  nose.tools.eq_(proc('102020002'), 0x4) #0xd #missing from bob
  nose.tools.eq_(proc('100020002'), 0x0) #non-uniform(2) => 0x0
  nose.tools.eq_(proc('102000200'), 0x0) #non-uniform(2) => 0x0
  nose.tools.eq_(proc('102020202'), 0x5) #0xf
  nose.tools.eq_(op.max_label, 0x6) # this is set to 6!

def test_vanilla_4p1r_toaverage():
  op = bob.ip.LBP(4,1,False,True)
  proc = Processor(op, generate_3x3_image, (1,1), 3)
  nose.tools.eq_(proc('100000000'), 0x0) #average is 0.2
  nose.tools.eq_(proc('102000000'), 0x8) #average is 0.6
  nose.tools.eq_(proc('100020000'), 0x4) #average is 0.6
  nose.tools.eq_(proc('100000200'), 0x2) #average is 0.6
  nose.tools.eq_(proc('100000002'), 0x1) #average is 0.6
  nose.tools.eq_(proc('102020000'), 0xc) #average is 1.0
  nose.tools.eq_(proc('100020200'), 0x6) #average is 1.0
  nose.tools.eq_(proc('100000202'), 0x3) #average is 1.0
  nose.tools.eq_(proc('102000002'), 0x9) #average is 1.0
  nose.tools.eq_(proc('102020200'), 0xe) #average is 1.4
  nose.tools.eq_(proc('100020202'), 0x7) #average is 1.4
  nose.tools.eq_(proc('102000202'), 0xb) #average is 1.4
  nose.tools.eq_(proc('102020002'), 0xd) #average is 1.4
  nose.tools.eq_(proc('100020002'), 0x5) #average is 1.0
  nose.tools.eq_(proc('102000200'), 0xa) #average is 1.0
  nose.tools.eq_(proc('102020202'), 0xf) #average is 1.8
  nose.tools.eq_(op.max_label, 0x10) # this is set to 16!

def test_vanilla_8p1r():
  op = bob.ip.LBP(8,1)
  proc = Processor(op, generate_3x3_image, (1,1), 3)
  for i in range(256):
    v = ('1%8s' % bin(i, 2)).replace(' ', '0')
    nose.tools.eq_(proc(v), i)


def test_rotinvariant_8p1r():
  op = bob.ip.LBP(8,1,False,False,False,False,True)
  proc = Processor(op, generate_3x3_image, (1,1), 3)
  table = calculate_lbp8r_rotinvariant_table()
  for i in range(256):
    v = ('1%8s' % bin(i, 2)).replace(' ', '0')
    nose.tools.eq_(proc(v), table[i])

def test_u2_8p1r():
  op = bob.ip.LBP(8,1,False,False,False,True,False)
  proc = Processor(op, generate_3x3_image, (1,1), 3)
  table = calculate_lbp8r_u2_table()
  values = []
  for i in range(256):
    v = ('1%8s' % bin(i, 2)).replace(' ', '0')
    values.append(proc(v))
    # just check that the zeros are good
    if not table[i] and i: nose.tools.eq_(values[-1], 0)
    if values[-1] and i:
      nose.tools.eq_(bool(table[i]), True)
  nose.tools.eq_(len(set(values)), len(set(table))+1)

def test_riu2_8p1r():
  op = bob.ip.LBP(8,1,False,False,False,True,True)
  proc = Processor(op, generate_3x3_image, (1,1), 3)
  table = calculate_lbp8r_riu2_table()
  values = []
  for i in range(256):
    v = ('1%8s' % bin(i, 2)).replace(' ', '0')
    values.append(proc(v))
    # just check that the zeros are good
    if not table[i] and i: nose.tools.eq_(values[-1], 0)
    if values[-1] and i:
      nose.tools.eq_(bool(table[i]), True)
  nose.tools.eq_(len(set(values)), len(set(table))+1)

def test_shape():
  lbp = bob.ip.LBP(8)
  image = numpy.ndarray((3,3), dtype='uint8')
  sh = lbp.get_lbp_shape(image)
  nose.tools.eq_(sh, (1,1))

  lbp = bob.ip.LBP(8, border_handling=bob.ip.LBPBorderHandling.WRAP)
  sh = lbp.get_lbp_shape(image)
  nose.tools.eq_(sh, (3,3))

def test_u2_16p1r():
  op = bob.ip.LBP(16, 1, True, False, False, True, False)
  values = [207, 24, 40, 36, 167, 230, 71, 247, 107, 9, 32, 139, 244, 233, 216, 232, 244, 123, 202, 238, 161, 246, 204, 244, 173]
  res = numpy.array(((214, 1, 122), (0, 4, 32), (12, 242, 178)), dtype=int)

  proc1 = Processor(op, generate_5x5_image, (1,1), 5); nose.tools.eq_(proc1(values), res[0,0])
  proc2 = Processor(op, generate_5x5_image, (2,1), 5); nose.tools.eq_(proc2(values), res[0,1])
  proc3 = Processor(op, generate_5x5_image, (3,1), 5); nose.tools.eq_(proc3(values), res[0,2])
  proc4 = Processor(op, generate_5x5_image, (1,2), 5); nose.tools.eq_(proc4(values), res[1,0])
  proc5 = Processor(op, generate_5x5_image, (2,2), 5); nose.tools.eq_(proc5(values), res[1,1])
  proc6 = Processor(op, generate_5x5_image, (3,2), 5); nose.tools.eq_(proc6(values), res[1,2])
  proc7 = Processor(op, generate_5x5_image, (1,3), 5); nose.tools.eq_(proc7(values), res[2,0])
  proc8 = Processor(op, generate_5x5_image, (2,3), 5); nose.tools.eq_(proc8(values), res[2,1])
  proc9 = Processor(op, generate_5x5_image, (3,3), 5); nose.tools.eq_(proc9(values), res[2,2])

def test_u2_16p2r():
  op = bob.ip.LBP(16, 2, True, False, False, True, False)
  values = [207, 24, 40, 36, 167, 230, 71, 247, 107, 9, 32, 139, 244, 233, 216, 232, 244, 123, 202, 238, 161, 246, 204, 244, 173]
  res = numpy.ndarray((1,1), dtype=int)
  res[0,0]=1;
  proc1 = Processor(op, generate_5x5_image, (2,2), 5); nose.tools.eq_(proc1(values), res[0,0])

def test_riu2_16p1r():
  op = bob.ip.LBP(16, 1, True, False, False, True, True)
  values = [207, 24, 40, 36, 167, 230, 71, 247, 107, 9, 32, 139, 244, 233, 216, 232, 244, 123, 202, 238, 161, 246, 204, 244, 173]
  res = numpy.ndarray((3,3), dtype=int)
  res[0,0]=15; res[0,1]=1; res[0,2]=9;
  res[1,0]=0; res[1,1]=2; res[1,2]=3;
  res[2,0]=2; res[2,1]=17; res[2,2]=13;
  proc1 = Processor(op, generate_5x5_image, (1,1), 5); nose.tools.eq_(proc1(values), res[0,0])
  proc2 = Processor(op, generate_5x5_image, (2,1), 5); nose.tools.eq_(proc2(values), res[0,1])
  proc3 = Processor(op, generate_5x5_image, (3,1), 5); nose.tools.eq_(proc3(values), res[0,2])
  proc4 = Processor(op, generate_5x5_image, (1,2), 5); nose.tools.eq_(proc4(values), res[1,0])
  proc5 = Processor(op, generate_5x5_image, (2,2), 5); nose.tools.eq_(proc5(values), res[1,1])
  proc6 = Processor(op, generate_5x5_image, (3,2), 5); nose.tools.eq_(proc6(values), res[1,2])
  proc7 = Processor(op, generate_5x5_image, (1,3), 5); nose.tools.eq_(proc7(values), res[2,0])
  proc8 = Processor(op, generate_5x5_image, (2,3), 5); nose.tools.eq_(proc8(values), res[2,1])
  proc9 = Processor(op, generate_5x5_image, (3,3), 5); nose.tools.eq_(proc9(values), res[2,2])

def test_eLBP_8p1r():
  op = bob.ip.LBP(8, 1, False, False, False, False, False, bob.ip.ELBPType.REGULAR) # eLBP_type = 0,
  proc1 = Processor(op, generate_3x3_image, (1,1), 3)
  nose.tools.eq_(proc1('012345678'), 0xff) #0x0
  op = bob.ip.LBP(8, 1, False, True, False, False, False, bob.ip.ELBPType.REGULAR) # eLBP_type = 0, to_average=True for modified LBP (MCT)
  proc2 = Processor(op, generate_3x3_image, (1,1), 3)
  nose.tools.eq_(proc2('012345678'), 0x1f) #0x0
  op = bob.ip.LBP(8, 1, False, False, False, False, False, bob.ip.ELBPType.TRANSITIONAL) # eLBP_type=1, transitional LBP
  proc3 = Processor(op, generate_3x3_image, (1,1), 3)
  nose.tools.eq_(proc3('014725836'), 0x25) #0x0
  op = bob.ip.LBP(8, 1, False, False, False, False, False, bob.ip.ELBPType.DIRECTION_CODED) # eLBP_type=2, direction coded LBP
  proc4 = Processor(op, generate_3x3_image, (1,1), 3)
  nose.tools.eq_(proc4('014725836'), 0x5d) #0x0



def test_vanilla_4p1r_rectangle():
  #please note that the bob implementation of LBPs is slightly different
  #then that of the original LBP paper:
  # s(x) >= 0 => LBP digit = 1
  # s(x) <  0 => LBO digit = 0

  op = bob.ip.LBP(4)
  values = [3,5,12,1,3, 4,5,2,10,13, 14,0,10,3,1, 20,12,0,1,2, 14,12,1,3,7]

  op.radius  = 1
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 0);

  op.radius  = 2
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 9);


  op.radii  = (2,1)
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 8);


  op.radii  = (1,2)
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 1);


def test_vanilla_8p1r_rectangle():
  #please note that the bob implementation of LBPs is slightly different
  #then that of the original LBP paper:
  # s(x) >= 0 => LBP digit = 1
  # s(x) <  0 => LBO digit = 0

  op = bob.ip.LBP(8)
  values = [3,5,12,1,3, 4,5,2,10,13, 14,0,10,3,1, 20,12,0,1,2, 14,12,1,3,7]

  op.radius  = 1
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 34);

  op.radius  = 2
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 67);

  op.radii  = (2,1)
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 66);

  op.radii  = (1,2)
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 35);


def test_vanilla_8p1r_elipse():
  #please note that the bob implementation of LBPs is slightly different
  #then that of the original LBP paper:
  # s(x) >= 0 => LBP digit = 1
  # s(x) <  0 => LBO digit = 0

  op = bob.ip.LBP(8)
  op.circular = True
  values = [3,5,12,1,3, 4,5,2,10,13, 14,0,10,3,1, 20,12,0,1,2, 14,12,1,3,7]

  op.radius  = 1
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 0);

  op.radius  = 2
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 67);

  op.radii   = (2,1)
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 64);

  op.radii   = (1,2)
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 3);


def test_vanilla_16p1r_elipse():
  #please note that the bob implementation of LBPs is slightly different
  #then that of the original LBP paper:
  # s(x) >= 0 => LBP digit = 1
  # s(x) <  0 => LBO digit = 0

  op = bob.ip.LBP(16, circular=True)
  values = [3,5,12,1,3, 4,5,2,10,13, 14,0,10,3,1, 20,12,0,1,2, 14,12,1,3,7]

  op.radius  = 1
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 0);

  op.radius  = 2
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 8206);

  op.radii   = (2,1)
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 8192);

  op.radii   = (1,2)
  proc = Processor(op, generate_5x5_image, (2,2), 5)
  nose.tools.eq_(proc(values), 14);


def test_mb_lbp():
  # Tests multi-block LBP
  op = bob.ip.LBP(8, (2,1))
  nose.tools.eq_(op.block_size, (2,1))
  nose.tools.eq_(op.block_overlap, (0,0))

  values = numpy.array([[3,5,12], [1,3,4], [5,2,10], [13,14,0], [10,3,1], [20,12,0]], dtype=numpy.uint8)
  nose.tools.eq_(op.get_lbp_shape(values), (1,1))
  # get the multi-block code for this image
  nose.tools.eq_(op(values)[0,0], 0x23)

  # generate integral image
  ii = numpy.ndarray((7,4), dtype = numpy.uint16)
  bob.ip.integral(values, ii, True)
  nose.tools.eq_(op.get_lbp_shape(ii, True), (1,1))
  # get the multi-block code for this image
  nose.tools.eq_(op(ii, True)[0,0], 0x23)

  # test that all the other types of LBP still work
  op = bob.ip.LBP(8, (2,1), uniform=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x00)

  op = bob.ip.LBP(8, (2,1), rotation_invariant=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x0d)

  op = bob.ip.LBP(8, (2,1), uniform=True, rotation_invariant=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x0)

  op = bob.ip.LBP(8, (2,1), to_average=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x27)

  op = bob.ip.LBP(8, (2,1), to_average=True, add_average_bit=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x04f)

  op = bob.ip.LBP(8, (2,1), elbp_type=bob.ip.ELBPType.TRANSITIONAL)
  nose.tools.eq_(op(ii, True)[0,0], 0x33)

  op = bob.ip.LBP(8, (2,1), elbp_type=bob.ip.ELBPType.DIRECTION_CODED)
  nose.tools.eq_(op(ii, True)[0,0], 0x76)


def test_omb_lbp():
  # Tests overlapping multi-block LBP
  op = bob.ip.LBP(8, (3,3),(2,1))
  nose.tools.eq_(op.block_size, (3,3))
  nose.tools.eq_(op.block_overlap, (2,1))

  values = numpy.array([[0,0,1,1,1,2,2], [1,1,2,2,2,3,3], [1,1,3,3,3,5,5], [3,3,5,5,5,7,7], [5,5,7,7,7,9,9]], dtype=numpy.uint8)
  nose.tools.eq_(op.get_lbp_shape(values), (1,1))
  # get the offset multi-block code for this image
  nose.tools.eq_(op(values)[0,0], 0x1e)

  # generate integral image
  ii = numpy.ndarray((6,8), dtype = numpy.uint16)
  bob.ip.integral(values, ii, True)
  nose.tools.eq_(op.get_lbp_shape(ii, True), (1,1))
  # get the offset multi-block code for this image
  nose.tools.eq_(op(ii, True)[0,0], 0x1e)

  # test that all the other types of LBP still work
  op = bob.ip.LBP(8, (3,3), (2,1), uniform=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x1d)

  op = bob.ip.LBP(8, (3,3), (2,1), rotation_invariant=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x08)

  op = bob.ip.LBP(8, (3,3), (2,1), uniform=True, rotation_invariant=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x05)

  op = bob.ip.LBP(8, (3,3), (2,1), to_average=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x1e)

  op = bob.ip.LBP(8, (3,3), (2,1), to_average=True, add_average_bit=True)
  nose.tools.eq_(op(ii, True)[0,0], 0x3c)

  op = bob.ip.LBP(8, (3,3), (2,1), elbp_type=bob.ip.ELBPType.TRANSITIONAL)
  nose.tools.eq_(op(ii, True)[0,0], 0x0f)

  op = bob.ip.LBP(8, (3,3), (2,1), elbp_type=bob.ip.ELBPType.DIRECTION_CODED)
  nose.tools.eq_(op(ii, True)[0,0], 0x0a)


def test_io():
  # Checks that the IO functionality of LBP works
  test_file = bob.test.utils.datafile("LBP.hdf5", "bob.ip.test")
  temp_file = bob.test.utils.temporary_filename()

  # create file
  lbp1 = bob.ip.LBP(8, (2,3), elbp_type=bob.ip.ELBPType.TRANSITIONAL, to_average=True, add_average_bit=True)
  lbp2 = bob.ip.LBP(16, 4., 2., uniform=True, rotation_invariant=True, circular=True)

  # re-generate the reference file, if wanted
  f = bob.io.HDF5File(temp_file, 'w')
  f.create_group("LBP1")
  f.create_group("LBP2")
  f.cd("/LBP1")
  lbp1.save(f)
  f.cd("/LBP2")
  lbp2.save(f)
  del f

  # load the file again
  f = bob.io.HDF5File(temp_file)
  f.cd("/LBP1")
  read1 = bob.ip.LBP(f)
  f.cd("/LBP2")
  read2 = bob.ip.LBP(f)
  del f

  # assert that the created and the read object are identical
  assert lbp1 == read1
  assert lbp2 == read2

  # load the reference file
  f = bob.io.HDF5File(test_file)
  f.cd("/LBP1")
  ref1 = bob.ip.LBP(f)
  f.cd("/LBP2")
  ref2 = bob.ip.LBP(f)
  del f

  # assert that the lbp objects and the reference ones are identical
  assert lbp1 == ref1
  assert lbp2 == ref2
  assert read1 == ref1
  assert read2 == ref2



"""
" All planes are p=4, r=1, non uniform pattern and non RI
"""
def test_vanilla_4p1r_4p1r_4p1r():
  lbp4R_XY = bob.ip.LBP(4, radius=1.0, circular=False, uniform=False, rotation_invariant=False)
  lbp4R_XT = bob.ip.LBP(4, radius=1.0, circular=False, uniform=False, rotation_invariant=False)
  lbp4R_YT = bob.ip.LBP(4, radius=1.0, circular=False, uniform=False, rotation_invariant=False)

  op = bob.ip.LBPTop(lbp4R_XY,lbp4R_XT,lbp4R_YT)

  proc1 = ProcessorLBPTop(op, generate_NxMxM_image,img_size=3,n_frames=3)
  nose.tools.eq_(proc1(['000000000','111111111','222222222'],plane_index=0,operator_coordinates=(0,0,0)),0xf)
  nose.tools.eq_(proc1(['000000000','111111111','222222222'],plane_index=1,operator_coordinates=(0,0,0)),0x7)
  nose.tools.eq_(proc1(['000000000','111111111','222222222'],plane_index=2,operator_coordinates=(0,0,0)),0x7)


  proc2 = ProcessorLBPTop(op, generate_NxMxM_image,img_size=5,n_frames=5)
  values_5x5 = []
  values_5x5.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  values_5x5.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
  values_5x5.append([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
  values_5x5.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
  values_5x5.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

  nose.tools.eq_(proc2(values_5x5,plane_index=0,operator_coordinates=(0,0,0)),0xf)
  nose.tools.eq_(proc2(values_5x5,plane_index=1,operator_coordinates=(0,0,0)),0x7)
  nose.tools.eq_(proc2(values_5x5,plane_index=2,operator_coordinates=(0,0,0)),0x7)


