#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sun  4 Mar 20:06:14 2012 
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


"""Tests for libsvm training
"""

import os, sys
import unittest
import math
import bob
import numpy
import tempfile

def tempname(suffix, prefix='bobtest_'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

TEST_MACHINE_NO_PROBS = 'heart_no_probs.svmmodel' 

HEART_DATA = 'heart.svmdata' #13 inputs
HEART_MACHINE = 'heart.svmmodel' #supports probabilities
HEART_EXPECTED = 'heart.out' #expected probabilities

IRIS_DATA = 'iris.svmdata'
IRIS_MACHINE = 'iris.svmmodel'
IRIS_EXPECTED = 'iris.out' #expected probabilities

class SvmTrainingTest(unittest.TestCase):
  """Performs various SVM training tests."""

  def test01_initialization(self):

    # tests and examplifies some initialization parameters

    # all defaults
    trainer = bob.trainer.SVMTrainer()

  def test02_training(self):
   
    # For this example I'm using an SVM file because of convinience. You only
    # need to make sure you can gather the input into 2D double arrays in which
    # each array represents data from one class and each line on such array
    # contains a sample.
    f = bob.machine.SVMFile(HEART_DATA, 13)
    labels, data = f.read_all()
    neg = numpy.vstack([k for i,k in enumerate(data) if labels[i] < 0])
    pos = numpy.vstack([k for i,k in enumerate(data) if labels[i] > 0])

    # Data is also pre-scaled so features remain in the range between -1 and
    # +1. libsvm, apparently, suggests you do that for all features.

    trainer = bob.trainer.SVMTrainer()
    #machine = trainer.train((neg,pos))


# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(SvmTrainingTest)
