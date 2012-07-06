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
      'fold1': (3959, 1189, 601),
      'fold2': (3984, 1210, 555),
      'fold3': (4041, 1156, 552),
      'fold4': (4082, 1107, 560),
      'fold5': (4070, 1112, 567),
      'fold6': (4095, 1127, 527),
      'fold7': (4058, 1094, 597),
      'fold8': (4024, 1124, 601),
      'fold9': (3971, 1198, 580),
      'fold10': (3959, 1181, 609)
    }

  expected_models = {
      'view1': (3443, 853, 0), 
      'fold1': (5345, 916, 472),
      'fold2': (5333, 930, 462),
      'fold3': (5381, 934, 440),
      'fold4': (5434, 902, 459),
      'fold5': (5473, 899, 436),
      'fold6': (5467, 895, 441),
      'fold7': (5408, 877, 476),
      'fold8': (5360, 917, 462),
      'fold9': (5339, 938, 458),
      'fold10': (5367, 920, 458)
    }

  expected_restricted_training_images = { 
      'view1': 3443, 
      'fold1': 2267,
      'fold2': 2228,
      'fold3': 2234,
      'fold4': 2293,
      'fold5': 2341,
      'fold6': 2362,
      'fold7': 2334,
      'fold8': 2356,
      'fold9': 2368,
      'fold10': 2320
    }
  
  expected_unrestricted_files = {
      'view1': (9525, 1549, 0),
      'fold1': (8874, 2990, 1369),
      'fold2': (8714, 3152, 1367),
      'fold3': (9408, 2736, 1089),
      'fold4': (9453, 2456, 1324),
      'fold5': (9804, 2413, 1016),
      'fold6': (9727, 2340, 1166),
      'fold7': (9361, 2182, 1690),
      'fold8': (9155, 2856, 1222),
      'fold9': (9114, 2912, 1207),
      'fold10': (9021, 2429, 1783)
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
      self.assertEqual(len(db.files(protocol=e, groups='world', subworld='threefolds')), l)

    # check that the number of files for the restricted case are 7701 in each fold 
    for i in range(1,11):
      self.assertEqual(len(db.files(protocol='fold%d'%i, subworld='sevenfolds')), 7701)
    
      
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
    
    numbers = ((2200, 1000, 0), (4200, 1200, 600))
    
    # check the number of pairs
    index = 10
    for e in sorted(self.expected_models.iterkeys()):
      self.assertEqual(len(db.pairs(protocol=e, groups='world')), numbers[index > 0][0])
      self.assertEqual(len(db.pairs(protocol=e, groups='dev')), numbers[index > 0][1])
      self.assertEqual(len(db.pairs(protocol=e, groups='eval')), numbers[index > 0][2])
      index -= 1

  def test_unrestricted(self):
    """Tests the unrestricted configuration"""

    db = bob.db.lfw.Database()
    
    # check that the training files in the unrestricted configuration fit
    for e,l in self.expected_unrestricted_files.iteritems():
      self.assertEqual(len(db.files(protocol=e, groups='world', type='unrestricted')), l[0])
      self.assertEqual(len(db.files(protocol=e, groups='dev', type='unrestricted')), l[1])
      self.assertEqual(len(db.files(protocol=e, groups='eval', type='unrestricted')), l[2])
   
# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(LFWDatabaseTest)
