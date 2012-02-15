#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Fri May 27 15:47:40 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

import fileinput
from optparse import OptionParser

def main():

  usage = "usage: %prog [options] <input_files> "

  parser = OptionParser(usage)
  parser.set_description("Test a GMM model")

  parser.add_option("-m",
                    "--model",
                    dest="model",
                    help="Client model",
                    type="string",
                    default="train.hdf5")

  parser.add_option("-n",
                    "--noworld",
                    action="store_true",
                    dest="noworld",
                    default="False",
                    help="Do not use a world model")

  parser.add_option("-w",
                    "--world-model",
                    dest="world_model",
                    help="World model",
                    type="string",
                    default="wm.hdf5")

  (options, args) = parser.parse_args()

  filelist = []
  for line in fileinput.input(args):
    myfile = line.rstrip('\r\n')

    # Create data with only one file
    ar = bob.io.Arrayset(myfile)

    # Compute the number of blocks
    n_blocks = len(ar)

    # Load the gmm
    if not options.noworld:
      prior_gmm = bob.machine.GMMMachine(bob.io.HDF5File(options.world_model))
    gmm = bob.machine.GMMMachine(bob.io.HDF5File(options.model))

    # Compute the score
    scoreCL = 0.
    scoreWM = 0.
    for v in ar:
      scoreCL += gmm.forward(v.get())
      if not options.noworld:
        scoreWM += prior_gmm.forward(v.get())

    # scoreWM is equal to zero if noworld is not set
    score = (scoreCL - scoreWM) / n_blocks

    # Print the score
    print myfile + " " + str(score)

