#!/usr/bin/env python

import torch

import os, sys
import optparse
import math


def facenorm(image_file, pos_file, A_OUTPUT_DIR,
A_CROP_EYES_D, A_CROP_H, A_CROP_W, A_CROP_OH, A_CROP_OW):

  # Initialize cropper and destination array
  FEN = torch.ip.FaceEyesNorm(A_CROP_EYES_D, A_CROP_H, A_CROP_W, A_CROP_OH, A_CROP_OW)
  cropped_img = torch.core.array.float64_2(A_CROP_H, A_CROP_W)

  # Process one file
  img = torch.database.Array(image_file).get()
  
  # Display file processed
  print >> sys.stderr, "Crop: " + image_file

  # Extract the eye position
  fpos = open(pos_file)
  eyes_pos = fpos.readline().split()
  LW = int(eyes_pos[0])
  LH = int(eyes_pos[1])
  RW = int(eyes_pos[2])
  RH = int(eyes_pos[3])

  # Call the face normalizer
  FEN(img, cropped_img, LH, LW, RH, RW)

  output_file = os.path.join(A_OUTPUT_DIR, os.path.splitext(os.path.basename(image_file))[0] + '.hdf5')

  # Save the outpu file
  torch.database.Array(cropped_img).save(output_file)
  print output_file

  # Close the .pos file
  fpos.close()


import fileinput
from optparse import OptionParser

usage = "usage: %prog [options] <image_filelist> <pos_filelist>"

parser = OptionParser(usage)
parser.set_description("Face normalisation")

parser.add_option("-o",
                  "--output-dir",
                  dest="output_dir",
                  help="Output directory",
                  type="string",
                  default="facenorm")
parser.add_option("-d",
                  "--crop_eyes_d",
                  dest="crop_eyes_d",
                  help="",
                  type="int",
                  default=33)
parser.add_option("-g",
                  "--crop_eyes_h",
                  dest="crop_eyes_h",
                  help="",
                  type="int",
                  default=80)
parser.add_option("-w",
                  "--crop_eyes_w",
                  dest="crop_eyes_w",
                  help="",
                  type="int",
                  default=64)
parser.add_option("-p",
                  "--crop_eyes_oh",
                  dest="crop_eyes_oh",
                  help="",
                  type="int",
                  default=16)
parser.add_option("-q",
                  "--crop_eyes_ow",
                  dest="crop_eyes_ow",
                  help="",
                  type="int",
                  default=32)
parser.add_option('--self-test',
                  action="store_true",
                  dest="test",
                  help=optparse.SUPPRESS_HELP,
                  default=False)
                  
(options, args) = parser.parse_args()

if options.test:
  if os.path.exists("/tmp/input.hdf5"):
    os.remove("/tmp/input.hdf5")
    
  if os.path.exists("/tmp/input.pos"):
    os.remove("/tmp/input.pos")
    
  options.output_dir = "/tmp/facenorm"
  array = torch.core.array.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                                  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                                  [30, 31, 32,  0, 34, 35,  0, 37, 38, 39],
                                  [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                                  [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                                  [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                                  [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                                  [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                                  [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]],
                                  'uint8')

  options.crop_eyes_d = 3
  options.crop_eyes_h = 7
  options.crop_eyes_w = 5
  options.crop_eyes_oh = 2
  options.crop_eyes_ow = 2
  torch.database.Array(array).save("/tmp/input.hdf5")

  f = open("/tmp/input.pos", 'w')
  f.write("%d %d %d %d\n" % (3,3 , 6,3))
  f.close()

  f = open("/tmp/input.lst", 'w')
  f.write("/tmp/input.hdf5\n")
  f.close()
  
  f = open("/tmp/inputpos.lst", 'w')
  f.write("/tmp/input.pos\n")
  f.close()

  args.append("/tmp/input.lst")
  args.append("/tmp/inputpos.lst")


if (len(args) != 2):
  parser.print_help()
  exit(1)

image_list = []
pos_list = []

# Read the list of images
file1 = open(args[0])
image_list = [l.strip().strip('\r\n') for l in file1.readlines() if l.strip()]
file1.close()

# Read the list of positions
file2 = open(args[1])
pos_list = [l.strip().strip('\r\n') for l in file2.readlines() if l.strip()]
file1.close()


# Check the number of files
if len(image_list) != len(pos_list):
  print >> sys.stderr, "File length differs (%d != %d)" % (len(image_list), len(pos_list))
  sys.exit(1)

# Create output directory
if not os.path.exists(options.output_dir):
  os.makedirs(options.output_dir)
  
for i in range(len(image_list)):
  # Do the face normalization
  facenorm(image_list[i], pos_list[i], options.output_dir,
  options.crop_eyes_d, options.crop_eyes_h, options.crop_eyes_w, options.crop_eyes_oh, options.crop_eyes_ow)
  

if options.test:
  os.remove("/tmp/input.hdf5")
  os.remove("/tmp/input.lst")
  os.remove("/tmp/input.pos")
  os.remove("/tmp/inputpos.lst")
  if not os.path.exists("/tmp/facenorm/input.hdf5"):
    sys.exit(1)
  else:
    os.remove("/tmp/facenorm/input.hdf5")
    try:
      os.rmdir("/tmp/facenorm")
    except:
      pass