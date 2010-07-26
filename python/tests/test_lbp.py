#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 23 Jul 2010 12:05:28 CEST 

"""Tests the LBP framework. Find attached to this test a table with expected
LBP codes.
"""

import unittest
import torch

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
  for k in range(256):
    retval.append(calculate_lbp8r_rotinvariant_value(k))
  return retval
   
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

if __name__ == '__main__':
  import os, sys
  sys.argv.append('-v')
  unittest.main()

