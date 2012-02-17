#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# André Anjos <andre.dos.anjos@gmail.com>
# Sat Dec 17 14:41:56 2011 +0100
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

"""Run tests on the libsvm machine infrastructure.
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

def load_expected(filename):
  """Loads libsvm's svm-predict output file with probabilities"""

  f = open(filename, 'rt')
  labels = [int(k) for k in f.readline().split()[1:]]

  predictions = []
  probabilities = []
  for k in f: #load the remaning lines
    s = k.split()
    predictions.append(int(s[0]))
    probabilities.append(numpy.array([float(c) for c in s[1:]], 'float64'))

  return tuple(labels), tuple(predictions), tuple(probabilities)

#extracted by running svm-predict.c on the heart_scale example data
expected_heart_predictions = (1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1,
    -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1,
    -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1,
    1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1,
    -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1,
    1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1,
    1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1,
    -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1,
    -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
    1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1,
    -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1,
    1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1,
    -1, -1, -1, -1, -1, -1, 1) 

expected_iris_predictions = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3)

class SvmTest(unittest.TestCase):
  """Performs various SVM tests."""

  def test01_can_load(self):

    machine = bob.machine.SupportVector(HEART_MACHINE)
    self.assertEqual(machine.shape, (13,1))
    self.assertEqual(machine.kernel_type, bob.machine.svm_kernel_type.RBF)
    self.assertEqual(machine.machine_type, bob.machine.svm_type.C_SVC)
    self.assertEqual(len(machine.labels), 2)
    self.assertTrue( -1 in machine.labels )
    self.assertTrue( +1 in machine.labels )
    self.assertTrue( abs(machine.gamma - 0.0769231) < 1e-6 )

  def test02_can_save(self):

    machine = bob.machine.SupportVector(HEART_MACHINE)
    tmp = tempname('.model')
    machine.save(tmp)
    del machine

    # make sure that the save machine is the same as before
    machine = bob.machine.SupportVector(tmp)
    self.assertEqual(machine.shape, (13,1))
    self.assertEqual(machine.kernel_type, bob.machine.svm_kernel_type.RBF)
    self.assertEqual(machine.machine_type, bob.machine.svm_type.C_SVC)
    self.assertEqual(len(machine.labels), 2)
    self.assertTrue( -1 in machine.labels )
    self.assertTrue( +1 in machine.labels )
    self.assertTrue( abs(machine.gamma - 0.0769231) < 1e-6 )

  def test03_data_loading(self):

    #tests if I can load data in libsvm format using SVMFile
    data = bob.machine.SVMFile(HEART_DATA, 13)
    self.assertEqual(data.shape, (13,))
    self.assertEqual(data.good(), True)
    self.assertEqual(data.fail(), False)
    self.assertEqual(data.eof(), False)

    #tries loading the data, one by one
    all_data = []
    all_labels = []
    while data.good():
      values = numpy.ndarray(data.shape, 'float64')
      label = data.read(values)
      if label:
        all_labels.append(label)
        all_data.append(values)
    all_labels = tuple(all_labels)

    self.assertEqual(len(all_data), len(all_labels))
    self.assertEqual(len(all_data), 270)

    #tries loading the data with numpy arrays allocated internally
    counter = 0
    data.reset()
    entry = data.read()
    while entry:
      self.assertEqual( entry[0], all_labels[counter] )
      self.assertTrue( numpy.array_equal(entry[1], all_data[counter]) )
      counter += 1
      entry = data.read()

    #tries loading the file all in a single shot
    data.reset()
    labels, data = data.read_all()
    self.assertEqual(labels, all_labels)
    for k, l in zip(data, all_data):
      self.assertTrue( numpy.array_equal(k, l) )

    #makes sure the first 3 examples are correctly read
    ex = []
    ex.append(numpy.array([0.708333 , 1, 1, -0.320755 , -0.105023 , -1, 1, 
      -0.419847 ,-1, -0.225806 ,0. ,1, -1], 'float64'))
    ex.append(numpy.array([0.583333, -1, 0.333333, -0.603774, 1, -1, 1,
      0.358779, -1, -0.483871, 0., -1, 1], 'float64'))
    ex.append(numpy.array([0.166667, 1, -0.333333, -0.433962, -0.383562, -1,
      -1, 0.0687023, -1, -0.903226, -1, -1, 1], 'float64'))
    ls = [+1, -1, +1]
    for k, (l, e) in enumerate(zip(ls, ex)):
      self.assertEqual( l, labels[k] )
      self.assertTrue ( numpy.array_equal(e, data[k]) )

  def test04_raises(self):

    #tests that the normal machine raises because probabilities are not
    #supported on that model
    machine = bob.machine.SupportVector(TEST_MACHINE_NO_PROBS)
    labels, data = bob.machine.SVMFile(HEART_DATA,
        machine.shape[0]).read_all()
    data = numpy.vstack(data)
    self.assertRaises(RuntimeError, machine.predictClassesAndProbabilities,
        data)

  def test05_correctness_heart(self):

    #tests the correctness of the libSVM bindings
    machine = bob.machine.SupportVector(HEART_MACHINE)
    labels, data = bob.machine.SVMFile(HEART_DATA,
        machine.shape[0]).read_all()
    data = numpy.vstack(data)

    pred_label = machine.predictClasses(data)

    self.assertEqual(pred_label, expected_heart_predictions)

    #finally, we test if the values also work fine.
    pred_lab_values = [machine.predictClassAndScores(k) for k in data]

    #tries the variant with multiple inputs
    pred_labels2, pred_scores2 = machine.predictClassesAndScores(data)
    self.assertEqual( expected_heart_predictions,  pred_labels2 )
    self.assertEqual( tuple([k[1] for k in pred_lab_values]), pred_scores2 )

    #tries to get the probabilities - note: for some reason, when getting
    #probabilities, the labels change, but notice the note bellow:

    # Note from the libSVM FAQ:
    # Q: Why using the -b option does not give me better accuracy? 
    # There is absolutely no reason the probability outputs guarantee you
    # better accuracy. The main purpose of this option is to provide you the
    # probability estimates, but not to boost prediction accuracy. From our
    # experience, after proper parameter selections, in general with and
    # without -b have similar accuracy. Occasionally there are some
    # differences. It is not recommended to compare the two under just a fixed
    # parameter set as more differences will be observed.
    all_labels, real_labels, real_probs = load_expected(HEART_EXPECTED)
    
    pred_labels, pred_probs = machine.predictClassesAndProbabilities(data)
    self.assertEqual(pred_labels, real_labels)
    self.assertTrue( numpy.all(abs(numpy.vstack(pred_probs) -
      numpy.vstack(real_probs)) < 1e-6) )

  def test06_correctness_iris(self):

    #same test as above, but with a 3-class problem.
    machine = bob.machine.SupportVector(IRIS_MACHINE)
    labels, data = bob.machine.SVMFile(IRIS_DATA, machine.shape[0]).read_all()
    data = numpy.vstack(data)

    pred_label = machine.predictClasses(data)

    self.assertEqual(pred_label, expected_iris_predictions)

    #finally, we test if the values also work fine.
    pred_lab_values = [machine.predictClassAndScores(k) for k in data]

    #tries the variant with multiple inputs
    pred_labels2, pred_scores2 = machine.predictClassesAndScores(data)
    self.assertEqual( expected_iris_predictions,  pred_labels2 )
    self.assertTrue( numpy.all(abs(numpy.vstack([k[1] for k in
      pred_lab_values]) - numpy.vstack(pred_scores2)) < 1e-20 ) )

    #tries to get the probabilities - note: for some reason, when getting
    #probabilities, the labels change, but notice the note bellow:

    all_labels, real_labels, real_probs = load_expected(IRIS_EXPECTED)
    
    pred_labels, pred_probs = machine.predictClassesAndProbabilities(data)
    self.assertEqual(pred_labels, real_labels)
    self.assertTrue( numpy.all(abs(numpy.vstack(pred_probs) -
      numpy.vstack(real_probs)) < 1e-6) )

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(SvmTest)
