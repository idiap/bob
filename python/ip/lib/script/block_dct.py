#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Fri May 27 15:47:40 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import bob
import os, sys
import optparse
import math
import numpy

def normalize_blocks(src):
  for i in range(src.shape[0]):
    block = src[i, :, :]
    mean = block.mean()
    std = ((block - mean) ** 2).sum() / block.size
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[i, :, :] = (block - mean) / std
    
def normalize_dct(src):
  for i in range(src.shape[1]):
    col = src[:, i]
    mean = col.mean()
    std = ((col - mean) ** 2).sum() / col.size
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[:, i] = (col - mean) / std


def dctfeatures(line, A_OUTPUT_DIR, A_OUTPUT_EXTENSION,
    A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, A_N_DCT_COEF,
    norm_before, norm_after, add_xy):
  
  # Display file processed
  print >> sys.stderr, "DCT: " + line
  
  # Process one file
  prep = bob.io.Array(line).get().astype('float64')

  blockShape = bob.ip.get_block_shape(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  blocks = numpy.ndarray(blockShape, 'float64')
  bob.ip.block(prep, blocks, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)

  if norm_before:
    normalize_blocks(blocks)

  if add_xy:
    real_DCT_coef = A_N_DCT_COEF - 2
  else:
    real_DCT_coef = A_N_DCT_COEF

  
  # Initialize cropper and destination array
  DCTF = bob.ip.DCTFeatures(A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, real_DCT_coef)
  
  # Call the preprocessing algorithm
  dct_blocks = DCTF(blocks)

  n_blocks = blockShape[0]

  dct_blocks_min = 0
  dct_blocks_max = A_N_DCT_COEF
  TMP_tensor_min = 0
  TMP_tensor_max = A_N_DCT_COEF

  if norm_before:
    dct_blocks_min += 1
    TMP_tensor_max -= 1

  if add_xy:
    dct_blocks_max -= 2
    TMP_tensor_min += 2
  
  TMP_tensor = numpy.ndarray((n_blocks, TMP_tensor_max), 'float64')
  
  nBlocks = bob.ip.get_n_blocks(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  for by in range(nBlocks[0]):
    for bx in range(nBlocks[1]):
      bi = bx + by * nBlocks[1]
      if add_xy:
        TMP_tensor[bi, 0] = bx
        TMP_tensor[bi, 1] = by
      
      TMP_tensor[bi, TMP_tensor_min:TMP_tensor_max] = dct_blocks[bi, dct_blocks_min:dct_blocks_max]

  if norm_after:
    normalize_dct(TMP_tensor)

  output_file = os.path.join(A_OUTPUT_DIR, os.path.splitext(os.path.basename(line))[0] + ".hdf5")
  bob.io.Array(TMP_tensor).save(output_file)

  print os.path.join(output_file)
  

import fileinput
from optparse import OptionParser

def main():

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
                    help="Normalize each block before DCT\n" +
                         "Warning: If you use this option, the first " +
                         "coefficient of the DCT is always 0 and is " +
                         "removed for the results. So if you choose n DCT " +
                         "coefficients, the results have (n-1) values.",
                    action="store_true",
                    default=False)
  parser.add_option("-a",
                    "--norm-after",
                    dest="norm_after",
                    help="Normalize the DCT coefficients",
                    action="store_true",
                    default=False)
  parser.add_option("-x",
                    "--add-xy",
                    dest="add_xy",
                    help="Add (x,y) block location to the DCT coefficients",
                    action="store_true",
                    default=False)
  parser.add_option('--self-test',
                    action="store_true",
                    dest="test",
                    help=optparse.SUPPRESS_HELP,
                    default=False)

  (options, args) = parser.parse_args()

  if options.test:
    array = numpy.array([[[ 8,  2], [ 2,  8] ]], 'float64')

    array_ref = numpy.array([[[ 1, -1], [-1,  1] ]], 'float64')
    normalize_blocks(array)
    if not (array == array_ref).all():
      print "Problem with normalize_blocks"
      sys.exit(1)
    
    array = numpy.array([[ 2,  8], [ 8,  2], [ 2,  8], [ 8,  2] ], 'float64')
    array_ref = numpy.array([[-1,  1], [ 1, -1], [-1,  1], [ 1, -1] ], 'float64')
    normalize_dct(array)
    if not (array == array_ref).all():
      print "Problem with normalize_dct"
      sys.exit(1)

    if os.path.exists("/tmp/input.hdf5"):
      os.remove("/tmp/input.hdf5")
    options.output_dir = "/tmp/blockDCT"
    array = numpy.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
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

    bob.io.Array(array).save("/tmp/input.hdf5")

    f = open("/tmp/input.lst", 'w')
    f.write("/tmp/input.hdf5\n")
    f.close()
    
    args.append("/tmp/input.lst")
    options.add_xy = True

  # Create output directory
  if not os.path.exists(options.output_dir):
    os.makedirs(options.output_dir)

  for line in fileinput.input(args):
    # Compute the dct of all files
    dctfeatures(line.rstrip('\r\n').strip(), options.output_dir, "", options.block_h, options.block_w,
    options.overlap_h, options.overlap_w, options.n_dct_coef,
    options.norm_before, options.norm_after, options.add_xy)
    

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
    
