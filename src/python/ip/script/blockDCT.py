#!/usr/bin/env python

import torch
import os, sys
import optparse
import math

def normalizeBlocks(src):
  for i in range(src.extent(0)):
    block = src[i, :, :]
    mean = torch.core.array.float64_2.mean(block)
    std = torch.core.array.float64_2.sum((block - mean) ** 2) / block.size()
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[i, :, :] = (block - mean) / std
    
def normalizeDCT(src):
  for i in range(src.extent(1)):
    col = src[:, i]
    mean = torch.core.array.float64_1.mean(col)
    std = torch.core.array.float64_1.sum((col - mean) ** 2) / col.size()
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[:, i] = (col - mean) / std


def dctfeatures(line, A_OUTPUT_DIR, A_OUTPUT_EXTENSION,
    A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, A_N_DCT_COEF,
    norm_before, norm_after):
  
  # Display file processed
  print >> sys.stderr, "DCT: " + line
  
  # Process one file
  prep = torch.io.Array(line).get().cast('float64')

  blockShape = torch.ip.getBlockShape(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  blocks = torch.core.array.float64_3(blockShape)
  torch.ip.block(prep, blocks, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)

  if norm_before:
    normalizeBlocks(blocks)
  
  # Initialize cropper and destination array
  DCTF = torch.ip.DCTFeatures(A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, A_N_DCT_COEF - 2)
  
  # Call the preprocessing algorithm
  dct_blocks = DCTF(blocks)
  
  n_blocks = blockShape[0]
  TMP_tensor = torch.core.array.float64_2(n_blocks, A_N_DCT_COEF)
  
  nBlocks = torch.ip.getNBlocks(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  for bx in range(nBlocks[0]):
    for by in range(nBlocks[1]):
      bi = bx + by * nBlocks[0]
      TMP_tensor[bi, 0] = bx
      TMP_tensor[bi, 1] = by
      for j in range(A_N_DCT_COEF-2):
          TMP_tensor[bi,j+2] = dct_blocks[bi, j]

  if norm_after:
    normalizeDCT(TMP_tensor)

  output_file = os.path.join(A_OUTPUT_DIR, os.path.splitext(os.path.basename(line))[0] + ".hdf5")
  torch.io.Array(TMP_tensor).save(output_file)

  print os.path.join(output_file)
  

import fileinput
from optparse import OptionParser

usage = "usage: %prog [options] <input_files> "

parser = OptionParser(usage)
parser.set_description("Extract DCT features by block and add x y coordinates of the block to the DCT coefficients")

parser.add_option("-o",
                  "--output-dir",
                  dest="output_dir",
                  help="Output directory",
                  type="string",
                  default="blockDCT")
parser.add_option("-c",
                  "--block-h",
                  dest="block_h",
                  help="",
                  type="int",
                  default=8)
parser.add_option("-w",
                  "--block-w",
                  dest="block_w",
                  help="",
                  type="int",
                  default=8)
parser.add_option("-p",
                  "--overlap-h",
                  dest="overlap_h",
                  help="",
                  type="int",
                  default=0)
parser.add_option("-q",
                  "--overlap-w",
                  dest="overlap_w",
                  help="",
                  type="int",
                  default=0)
parser.add_option("-n",
                  "--n-dct-coef",
                  dest="n_dct_coef",
                  help="",
                  type="int",
                  default=15)
parser.add_option("-b",
                  "--norm-before",
                  dest="norm_before",
                  help="Normalize each block before DCT",
                  action="store_true",
                  default=False)
parser.add_option("-a",
                  "--norm-after",
                  dest="norm_after",
                  help="Normalize the DCT coefficients",
                  action="store_true",
                  default=False)
parser.add_option('--self-test',
                  action="store_true",
                  dest="test",
                  help=optparse.SUPPRESS_HELP,
                  default=False)

(options, args) = parser.parse_args()

if options.test:
  array = torch.core.array.array([[[ 8,  2],
                                   [ 2,  8]
                                   ]],
                                  'float64')

  array_ref = torch.core.array.array([[[ 1, -1],
                                       [-1,  1]
                                       ]],
                                       'float64')
  normalizeBlocks(array)
  if not (array == array_ref).all():
    print "Problem with normalizeBlocks"
    sys.exit(1)
  
  array = torch.core.array.array([[ 2,  8],
                                  [ 8,  2],
                                  [ 2,  8],
                                  [ 8,  2]
                                  ],
                                  'float64')


  array_ref = torch.core.array.array([[-1,  1],
                                      [ 1, -1],
                                      [-1,  1],
                                      [ 1, -1]
                                      ],
                                      'float64')
  normalizeDCT(array)
  if not (array == array_ref).all():
    print "Problem with normalizeDCT"
    sys.exit(1)

  if os.path.exists("/tmp/input.hdf5"):
    os.remove("/tmp/input.hdf5")
  options.output_dir = "/tmp/blockDCT"
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

  torch.io.Array(array).save("/tmp/input.hdf5")

  f = open("/tmp/input.lst", 'w')
  f.write("/tmp/input.hdf5\n")
  f.close()
  
  args.append("/tmp/input.lst")

# Create output directory
if not os.path.exists(options.output_dir):
  os.makedirs(options.output_dir)

for line in fileinput.input(args):
  # Compute the dct of all files
  dctfeatures(line.rstrip('\r\n').strip(), options.output_dir, "", options.block_h, options.block_w,
  options.overlap_h, options.overlap_w, options.n_dct_coef,
  options.norm_before, options.norm_after)
  

if options.test:
  os.remove("/tmp/input.hdf5")
  os.remove("/tmp/input.lst")
  
  if not os.path.exists("/tmp/blockDCT/input.hdf5"):
    sys.exit(1)
  else:
    os.remove("/tmp/blockDCT/input.hdf5")
    try:
      os.rmdir("/tmp/blockDCT")
    except:
      pass
  