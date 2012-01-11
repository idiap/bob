#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 25 Aug 2010 12:33:13 CEST 

"""Test all ip image filters.
"""

import math
import os, sys
import unittest
import bob
import numpy

# face data
LH = 120 # Left eye height
LW = 147 # Left eye width
RH = 90  # Right eye height
RW = 213 # Right eye width

GOAL_EYE_DISTANCE = 30

class FilterNewTest(unittest.TestCase):
  """Performs various combined filter tests."""

  def test01_shiftColorImage(self):
    print ""

    img = bob.io.Array(os.path.join('data', 'faceextract', 'test_001.png'))
    A = img.get()
    B = A.copy()

    delta_h = 100
    delta_w = 100

    # shift to center
    bob.ip.shift(A, B, delta_h, delta_w);

    # save image
    bob.io.Array(B).save(os.path.join('data', 'faceextract', 'test_001.shift.png'));

  def test02_shiftToCenterBlue(self):
    print ""

    img = bob.io.Array(os.path.join('data', 'faceextract', 'test_001.png'))
    A = img.get()
    B = A.copy()

    # shift to center
    bob.ip.shiftToCenterOfPoints(A, B, LH, LW, RH, RW)

    # save image
    bob.io.Array(B).save(os.path.join('data', 'faceextract', 'test_001.blue.answer.png'));

  def test03_shiftToCenterBlue_And_LevelOut(self):
    print ""

    img = bob.io.Array(os.path.join('data', 'faceextract', 'test_001.gray.png'))
    A = img.get()[1,:,:]
    B = A.copy()

    # shift to center
    bob.ip.shiftToCenterOfPoints(A, B, LH, LW, RH, RW)

    # rotate
    angle = bob.ip.getAngleToHorizontal(LH, LW, RH, RW)
    shape = bob.ip.getShapeRotated(B, angle)
    C = B.copy()
    C.resize(shape)
    bob.ip.rotate(B, C, angle)
    
    # save image
    bob.io.Array(C).save(os.path.join('data', 'faceextract', 'test_001.blue.level.answer.png'));

  def test04_geoNormBlue(self):
    print ""

    # read up image
    img = bob.io.Array(os.path.join('data', 'faceextract', 'test_001.gray.png'))
    A = img.get()[1,:,:]
    B = A.copy()

    # shift to center
    bob.ip.shiftToCenterOfPoints(A, B, LH, LW, RH, RW)

    # rotate
    angle = bob.ip.getRotateAngleToLevelOutHorizontal(LH, LW, RH, RW)
    shape = bob.ip.getShapeRotated(B, angle)
    C = B.copy()
    C.resize(shape)
    bob.ip.rotate(B, C, angle)

    # normalise
    previous_eye_distance = math.sqrt((RH - LH) * (RH - LH) + (RW - LW) * (RW - LW))
    print previous_eye_distance

    scale_factor = GOAL_EYE_DISTANCE / previous_eye_distance;

    #
    D = bob.ip.scaleAs(C, scale_factor)
    bob.ip.scale(C, D)

  def test05_geoNormFace(self):
    print ""

    # read up image
    img = bob.io.Array(os.path.join('data', 'faceextract', 'test-faces.jpg'))
    A = img.get()[1,:,:]

    # read up the eye coordinates
    f = open(os.path.join('data', 'faceextract', 'test-faces.txt'));
    coord = f.readline().split('\n')[0].split(' ')
    print coord

    LH = int(coord[0])
    LW = int(coord[1])
    RH = int(coord[2])
    RW = int(coord[3])

    # shift to center
    B = A.copy()
    bob.ip.shiftToCenterOfPoints(A, B, LH, LW, RH, RW)

    # rotate
    angle = bob.ip.getRotateAngleToLevelOutHorizontal(LH, LW, RH, RW)
    shape = bob.ip.getShapeRotated(B, angle)
    C = B.copy()
    C.resize(shape)
    bob.ip.rotate(B, C, angle)

    # normalise
    previous_eye_distance = math.sqrt((RH - LH) * (RH - LH) + (RW - LW) * (RW - LW))
    print previous_eye_distance

    scale_factor = GOAL_EYE_DISTANCE / previous_eye_distance

    #
    D = bob.ip.scaleAs(C, scale_factor)
    bob.ip.scale(C, D)
    bob.io.Array(D).save(os.path.join('data', 'faceextract', 'test-faces.1.jpg'));

    
    # crop face
    E = bob.core.array.uint8_2(100, 100)
    bob.ip.cropFace(D, E, 30)
    bob.io.Array(E).save(os.path.join('data', 'faceextract', 'test-faces.E.jpg'));


if __name__ == '__main__':
  sys.argv.append('-v')
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()

