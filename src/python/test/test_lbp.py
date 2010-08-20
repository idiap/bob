#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 23 Jul 2010 12:05:28 CEST 

"""Tests the LBP framework. Find attached to this test a table with expected
LBP codes.
"""

import unittest
import torch
import math

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
  #remember: image.set(y, x, val) and NOT image.set(x, y, val)!
  image.set(1, 1, 0, int(values[0]))
  image.set(0, 0, 0, int(values[1]))
  image.set(0, 1, 0, int(values[2]))
  image.set(0, 2, 0, int(values[3]))
  image.set(1, 2, 0, int(values[4]))
  image.set(2, 2, 0, int(values[5]))
  image.set(2, 1, 0, int(values[6]))
  image.set(2, 0, 0, int(values[7]))
  image.set(1, 0, 0, int(values[8]))
  #image.save('img_%s.pgm' % values)

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
  map = {} #map into torch values
  last = 1
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
  map = {} #map into torch values
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
  y1 = ((xh - x) * image.get(yl, xl)) + ((x - xl) * image.get(yl, xh))
  y2 = ((xh - x) * image.get(yh, xl)) + ((x - xl) * image.get(yh, xh))
  retval = ((yh - y) * y1) + ((y - yl) * y2)

class Processor:

  def __init__(self, operator, generator, center):
    self.operator = operator
    self.operator.setXY(center[0], center[1])
    self.generator = generator
    self.image = torch.ip.Image(3, 3, 1)

  def __call__(self, value):
    image = self.generator(self.image, value)
    self.operator.process(self.image)
    return self.operator.value

def bin(s, m=1):
  """Converts the number s into its binary representation (as a string)"""
  return str(m*s) if s<=1 else bin(s>>1, m) + str(m*(s&1))

class LBPTest(unittest.TestCase):
  """Performs various tests for the Torch::ipLBP and friends types."""
 
  def test01_vanilla_4p1r(self):
    op = torch.ip.ipLBP4R(1)
    proc = Processor(op, generate_3x3_image, (1,1))
    self.assertEqual(proc('011111111'), 0xf)
    #please note that the Torch implementation of LBPs is slightly different
    #then that of the original LBP paper:
    # paper:
    # s(x) >= 0 => LBP digit = 1
    # s(x) <  0 => LBP digit = 0
    # torch's:
    # s(x) >  0 => LBP digit = 1
    # s(x) <= 0 => LBO digit = 0
    self.assertEqual(proc('100000000'), 0x0)
    self.assertEqual(proc('102000000'), 0x8)
    self.assertEqual(proc('100020000'), 0x4)
    self.assertEqual(proc('100000200'), 0x2)
    self.assertEqual(proc('100000002'), 0x1)
    self.assertEqual(proc('102020000'), 0xc)
    self.assertEqual(proc('100020200'), 0x6)
    self.assertEqual(proc('100000202'), 0x3)
    self.assertEqual(proc('102000002'), 0x9)
    self.assertEqual(proc('102020200'), 0xe)
    self.assertEqual(proc('100020202'), 0x7)
    self.assertEqual(proc('102000202'), 0xb)
    self.assertEqual(proc('102020002'), 0xd)
    self.assertEqual(proc('100020002'), 0x5)
    self.assertEqual(proc('102000200'), 0xa)
    self.assertEqual(proc('102020202'), 0xf)
    #self.assertEqual(op.max_label, 0xf) this is set to 16!

  def test02_rotinvariant_4p1r(self):
    op = torch.ip.ipLBP4R(1)
    op.setBOption("RotInvariant", True)
    proc = Processor(op, generate_3x3_image, (1,1))
    #torch's implementation start labelling the patterns from 1
    self.assertEqual(proc('100000000'), 0x1) #0x0
    self.assertEqual(proc('102000000'), 0x2) #0x1
    self.assertEqual(proc('100020000'), 0x2) #0x1
    self.assertEqual(proc('100000200'), 0x2) #0x1
    self.assertEqual(proc('100000002'), 0x2) #0x1
    self.assertEqual(proc('102020000'), 0x3) #0x3
    self.assertEqual(proc('100020200'), 0x3) #0x3
    self.assertEqual(proc('100000202'), 0x3) #0x3
    self.assertEqual(proc('102000002'), 0x3) #0x3
    self.assertEqual(proc('102020200'), 0x4) #0x7
    self.assertEqual(proc('100020202'), 0x4) #0x7
    self.assertEqual(proc('102000202'), 0x4) #0x7
    self.assertEqual(proc('102020002'), 0x4) #0x7
    #torch has these labels wrong
    #self.assertEqual(proc('100020002'), 0x5) #0x5
    #self.assertEqual(proc('102000200'), 0x5) #0x5
    #self.assertEqual(proc('102020202'), 0x6) #0xf
    #self.assertEqual(op.max_label, 0xf) this is set to 16!

  def test03_u2_4p1r(self):
    op = torch.ip.ipLBP4R(1)
    op.setBOption("Uniform", True)
    proc = Processor(op, generate_3x3_image, (1,1))
    self.assertEqual(proc('100000000'), 0x1) #0x0
    self.assertEqual(proc('102000000'), 0x2) #0x8
    self.assertEqual(proc('100020000'), 0x3) #0x4
    self.assertEqual(proc('100000200'), 0x4) #0x2
    self.assertEqual(proc('100000002'), 0x5) #0x1
    self.assertEqual(proc('102020000'), 0x6) #0xc
    self.assertEqual(proc('100020200'), 0x7) #0x6
    self.assertEqual(proc('100000202'), 0x8) #0x3
    self.assertEqual(proc('102000002'), 0x9) #0x9 (by chance!)
    self.assertEqual(proc('102020200'), 0xa) #0xe
    self.assertEqual(proc('100020202'), 0xb) #0x7
    self.assertEqual(proc('102000202'), 0xc) #0xb
    self.assertEqual(proc('102020002'), 0xd) #0xd (by chance!)
    self.assertEqual(proc('100020002'), 0x0) #non-uniform(2) => 0x0
    self.assertEqual(proc('102000200'), 0x0) #non-uniform(2) => 0x0
    self.assertEqual(proc('102020202'), 0xe) #0xf

  def test04_rotinvariant_u2_4p1r(self):
    op = torch.ip.ipLBP4R(1)
    op.setBOption("Uniform", True)
    op.setBOption("RotInvariant", True)
    proc = Processor(op, generate_3x3_image, (1,1))
    self.assertEqual(proc('100000000'), 0x1) #0x0
    self.assertEqual(proc('102000000'), 0x2) #0x8
    self.assertEqual(proc('100020000'), 0x2) #0x4
    self.assertEqual(proc('100000200'), 0x2) #0x2
    self.assertEqual(proc('100000002'), 0x2) #0x1
    self.assertEqual(proc('102020000'), 0x3) #0xc
    self.assertEqual(proc('100020200'), 0x3) #0x6
    self.assertEqual(proc('100000202'), 0x3) #0x3
    #self.assertEqual(proc('102000002'), 0x3) #0x9 #missing from torch
    self.assertEqual(proc('102020200'), 0x4) #0xe
    self.assertEqual(proc('100020202'), 0x4) #0x7
    #self.assertEqual(proc('102000202'), 0x4) #0xb #missing from torch
    #self.assertEqual(proc('102020002'), 0x4) #0xd #missing from torch
    self.assertEqual(proc('100020002'), 0x0) #non-uniform(2) => 0x0
    self.assertEqual(proc('102000200'), 0x0) #non-uniform(2) => 0x0
    self.assertEqual(proc('102020202'), 0x5) #0xf

  def test05_vanilla_4p1r_toaverage(self):
    op = torch.ip.ipLBP4R(1)
    proc = Processor(op, generate_3x3_image, (1,1))
    op.setBOption("ToAverage", True)
    self.assertEqual(proc('100000000'), 0x0) #average is 0.2
    self.assertEqual(proc('102000000'), 0x8) #average is 0.6
    self.assertEqual(proc('100020000'), 0x4) #average is 0.6
    self.assertEqual(proc('100000200'), 0x2) #average is 0.6
    self.assertEqual(proc('100000002'), 0x1) #average is 0.6
    self.assertEqual(proc('102020000'), 0xc) #average is 1.0
    self.assertEqual(proc('100020200'), 0x6) #average is 1.0
    self.assertEqual(proc('100000202'), 0x3) #average is 1.0
    self.assertEqual(proc('102000002'), 0x9) #average is 1.0
    self.assertEqual(proc('102020200'), 0xe) #average is 1.4
    self.assertEqual(proc('100020202'), 0x7) #average is 1.4
    self.assertEqual(proc('102000202'), 0xb) #average is 1.4
    self.assertEqual(proc('102020002'), 0xd) #average is 1.4
    self.assertEqual(proc('100020002'), 0x5) #average is 1.0
    self.assertEqual(proc('102000200'), 0xa) #average is 1.0
    self.assertEqual(proc('102020202'), 0xf) #average is 1.8
    #self.assertEqual(op.max_label, 0xf) this is set to 16!

  def test06_vanilla_8p1r(self):
    op = torch.ip.ipLBP8R(1)
    proc = Processor(op, generate_3x3_image, (1,1))
    for i in range(256):
      v = ('1%8s' % bin(i, 2)).replace(' ', '0')
      self.assertEqual(proc(v), i)

  def test07_rotinvariant_8p1r(self):
    op = torch.ip.ipLBP8R(1)
    op.setBOption("RotInvariant", True)
    proc = Processor(op, generate_3x3_image, (1,1))
    table = calculate_lbp8r_rotinvariant_table()
    for i in range(256):
      v = ('1%8s' % bin(i, 2)).replace(' ', '0')
      self.assertEqual(proc(v), table[i])

  def test08_u2_8p1r(self):
    op = torch.ip.ipLBP8R(1)
    op.setBOption("Uniform", True)
    proc = Processor(op, generate_3x3_image, (1,1))
    table = calculate_lbp8r_u2_table()
    values = []
    for i in range(256):
      v = ('1%8s' % bin(i, 2)).replace(' ', '0')
      values.append(proc(v))
      # just check that the zeros are good
      if not table[i] and i: self.assertEqual(values[-1], 0)
      if values[-1] and i: 
        self.assertEqual(bool(table[i]), True)
    self.assertEqual(len(set(values)), len(set(table))+1)

  def test09_riu2_8p1r(self):
    op = torch.ip.ipLBP8R(1)
    op.setBOption("RotInvariant", True)
    op.setBOption("Uniform", True)
    proc = Processor(op, generate_3x3_image, (1,1))
    table = calculate_lbp8r_riu2_table()
    values = []
    for i in range(256):
      v = ('1%8s' % bin(i, 2)).replace(' ', '0')
      values.append(proc(v)) 
      # just check that the zeros are good
      if not table[i] and i: self.assertEqual(values[-1], 0)
      if values[-1] and i: 
        self.assertEqual(bool(table[i]), True)
    self.assertEqual(len(set(values)), len(set(table))+1)

if __name__ == '__main__':
  import os, sys
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  os.umask(0) # makes sure all files created are removeable by others
  unittest.main()

