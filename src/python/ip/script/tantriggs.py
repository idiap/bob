#!/usr/bin/env python

import torch
import os, sys
import optparse
import math

def tantriggs(line, A_OUTPUT_DIR, A_OUT_H, A_OUT_W,
  A_GAMMA, A_SIGMA0, A_SIGMA1, A_SIZE, A_THRESHOLD, A_ALPHA):

  # Initialize cropper and destination array
  TT = torch.ip.TanTriggs(A_GAMMA, A_SIGMA0, A_SIGMA1, A_SIZE, A_THRESHOLD, A_ALPHA)
  preprocessed_img = torch.core.array.float64_2(A_OUT_H, A_OUT_W)

  # Process one file
  img = torch.database.Array(line).get()
  
  # Display file processed
  print >> sys.stderr, "Tan and Triggs: " + line

  # Call the preprocessing algorithm
  TT(img, preprocessed_img)

  output_file = os.path.join(A_OUTPUT_DIR, os.path.splitext(os.path.basename(line))[0] + ".hdf5")

  # Save output file
  torch.database.Array(preprocessed_img).save(output_file)
  print output_file


import fileinput
from optparse import OptionParser

usage = "usage: %prog [options] <input_files> "

parser = OptionParser(usage)
parser.set_description("Tan and Triggs")

parser.add_option("-o",
                  "--output-dir",
                  dest="output_dir",
                  help="Output directory",
                  type="string",
                  default="tantriggs")
parser.add_option("-f",
                  "--out-h",
                  dest="out_h",
                  help="",
                  type="int",
                  default=64)
parser.add_option("-w",
                  "--out_w",
                  dest="out_w",
                  help="",
                  type="int",
                  default=80)
parser.add_option("-g",
                  "--gamma",
                  dest="gamma",
                  help="",
                  type="float",
                  default=0.2)
parser.add_option("-s",
                  "--sigma0",
                  dest="sigma0",
                  help="",
                  type="float",
                  default=1.)
parser.add_option("-u",
                  "--sigma1",
                  dest="sigma1",
                  help="",
                  type="float",
                  default=2.)
parser.add_option("-v",
                  "--size",
                  dest="size",
                  help="",
                  type="int",
                  default=5)
parser.add_option("-t",
                  "--threshold",
                  dest="threshold",
                  help="",
                  type="float",
                  default=10.)
parser.add_option("-a",
                  "--alpha",
                  dest="alpha",
                  help="",
                  type="float",
                  default=0.1)
parser.add_option('--self-test',
                  action="store_true",
                  dest="test",
                  help=optparse.SUPPRESS_HELP,
                  default=False)

(options, args) = parser.parse_args()

if options.test:
  if os.path.exists("/tmp/input.hdf5"):
    os.remove("/tmp/input.hdf5")
    
  options.output_dir = "/tmp/tantriggs"
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

  torch.database.Array(array).save("/tmp/input.hdf5")

  f = open("/tmp/input.lst", 'w')
  f.write("/tmp/input.hdf5\n")
  f.close()
  
  args.append("/tmp/input.lst")
  
# Create output directory
if not os.path.exists(options.output_dir):
  os.makedirs(options.output_dir)

for line in fileinput.input(args):
  # Compute Tan & Triggs
  tantriggs(line.rstrip('\r\n').strip(), options.output_dir, options.out_h, options.out_w,
  options.gamma, options.sigma0, options.sigma1, options.size, options.threshold, options.alpha)
  

if options.test:
  os.remove("/tmp/input.hdf5")
  os.remove("/tmp/input.lst")
  
  if not os.path.exists("/tmp/tantriggs/input.hdf5"):
    sys.exit(1)
  else:
    os.remove("/tmp/tantriggs/input.hdf5")
    try:
      os.rmdir("/tmp/tantriggs")
    except:
      pass
  
  