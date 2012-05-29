#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date: Thu May 24 10:41:42 CEST 2012
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

"""Checks for the Labeled Faces in the Wild (LFW) database.
"""

import os, sys
import unittest
import bob

class LFWDatabaseTest(unittest.TestCase):
  """Performs some tests on the AT&T database."""

  # expected numbers of clients
  expected_clients = {
      'view1': (4038, 1711, 0), 
      'fold1': (4539, 609, 601),
      'fold2': (4593, 601, 555),
      'fold3': (4642, 555, 552),
      'fold4': (4637, 552, 560),
      'fold5': (4622, 560, 567),
      'fold6': (4655, 567, 527),
      'fold7': (4625, 527, 597),
      'fold8': (4551, 597, 601),
      'fold9': (4568, 601, 580),
      'fold10': (4560, 580, 609)
    }

  expected_models = {
      'view1': (9525, 853, 0), 
      'fold1': (10081, 458, 472),
      'fold2': (10497, 472, 462),
      'fold3': (10777, 462, 440),
      'fold4': (10820, 440, 459),
      'fold5': (10893, 459, 436),
      'fold6': (11051, 436, 441),
      'fold7': (10377, 441, 476),
      'fold8': (10321, 476, 462),
      'fold9': (10804, 462, 458),
      'fold10': (10243, 458, 458)
    }
  
  expected_restricted_training_images = { 
      'view1': 3443, 
      'fold1': 6109,
      'fold2': 6118,
      'fold3': 6188,
      'fold4': 6210,
      'fold5': 6210,
      'fold6': 6221,
      'fold7': 6145,
      'fold8': 6103,
      'fold9': 6152,
      'fold10': 6152
    }

  def test_clients(self):
    """Tests if the clients() and models() functions work as expected"""
    db = bob.db.lfw.Database()
    
    # check the number of clients per protocol    
    for e,l in self.expected_clients.iteritems():
      self.assertEqual(len(db.clients(protocol=e, groups='world')), l[0])
      self.assertEqual(len(db.clients(protocol=e, groups='dev')), l[1])
      self.assertEqual(len(db.clients(protocol=e, groups='eval')), l[2])

    # check the number of models per protocol
    for e,l in self.expected_models.iteritems():
      self.assertEqual(len(db.models(protocol=e, groups='world')), l[0])
      self.assertEqual(len(db.models(protocol=e, groups='dev')), l[1])
      self.assertEqual(len(db.models(protocol=e, groups='eval')), l[2])


  def test_files(self):
    """Tests if the files() function returns the expected number and type of files"""
    
    db = bob.db.lfw.Database()

    # check that the files() function returns the same number of elements as the models() function does
    for e,l in self.expected_models.iteritems():
      self.assertEqual(len(db.files(protocol=e, groups='world')), l[0])
      self.assertEqual(len(db.files(protocol=e, groups='dev', purposes='enrol')), l[1])
      self.assertEqual(len(db.files(protocol=e, groups='eval', purposes='enrol')), l[2])
    
    # also check that the training files in the restricted configuration fit
    for e,l in self.expected_restricted_training_images.iteritems():
      self.assertEqual(len(db.files(protocol=e, groups='world', subworld='restricted')), l)

    # check that the number of files for the restricted case are 7701 in each fold 
    for i in range(1,11):
      self.assertEqual(len(db.files(protocol='fold%d'%i, subworld='restricted')), 7701)
    
      
    # check that the probe files sum up to 1000 (view1) or 600 (view2)
    for e in self.expected_models.iterkeys():
      expected_probe_count = len(db.pairs(protocol=e, groups='dev'))
      # count the probes for each model
      current_probe_count = 0
      for model_id in db.models(protocol=e, groups='dev'):
        current_probe_count += len(db.files(protocol=e, groups='dev', purposes='probe', model_ids = (model_id,)))
      # assure that the number of probes is equal to the number of pairs
      self.assertEqual(current_probe_count, expected_probe_count)
      
  def test_pairs(self):
    """Tests if the pairs() function returns the desired output"""
    
    db = bob.db.lfw.Database()
    
    numbers = ((2200, 1000, 0), (4800, 600, 600))
    
    # check the number of pairs
    index = 10
    for e in sorted(self.expected_models.iterkeys()):
      self.assertEqual(len(db.pairs(protocol=e, groups='world')), numbers[index > 0][0])
      self.assertEqual(len(db.pairs(protocol=e, groups='dev')), numbers[index > 0][1])
      self.assertEqual(len(db.pairs(protocol=e, groups='eval')), numbers[index > 0][2])
      index -= 1

   
# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(LFWDatabaseTest)
