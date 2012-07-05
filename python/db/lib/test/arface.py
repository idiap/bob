#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Wed Jul  4 14:12:51 CEST 2012
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

"""Checks for the AR face database.
"""

import os, sys
import unittest
import bob

class ARfaceDatabaseTest(unittest.TestCase):
  """Performs some tests on the AR face database."""

  def test_clients(self):
    db = bob.db.arface.Database()
    
    # test that the expected number of clients is returned
    self.assertEqual(len(db.clients()), 136)
    self.assertEqual(len(db.clients(genders='m')), 76)
    self.assertEqual(len(db.clients(genders='w')), 60)
    self.assertEqual(len(db.clients(groups='world')), 50)
    self.assertEqual(len(db.clients(groups='dev')), 43)
    self.assertEqual(len(db.clients(groups='eval')), 43)
    self.assertEqual(len(db.clients(groups='dev', genders='m')), 24)
    self.assertEqual(len(db.clients(groups='eval', genders='m')), 24)
    
    self.assertEqual(db.clients(), db.models())
    

  def test_files(self):
    db = bob.db.arface.Database()
    
    # test that the files() function returns reasonable numbers of files
    self.assertEqual(len(db.files()), 3312)
    
    # number of world files for the two genders
    self.assertEqual(len(db.files(groups='world')), 1076)
    self.assertEqual(len(db.files(groups='world', genders='m')), 583)
    self.assertEqual(len(db.files(groups='world', genders='w')), 493)

    # number of world files might differ for some protocols 
    # since all identities that did not contain all files went into the world set
    self.assertEqual(len(db.files(groups='world', protocol='expression')), 330)
    self.assertEqual(len(db.files(groups='world', protocol='illumination')), 329)
    self.assertEqual(len(db.files(groups='world', protocol='occlusion')), 247)
    self.assertEqual(len(db.files(groups='world', protocol='occlusion_and_illumination')), 413)

    
    for g in ['dev', 'eval']:
      # assert that each dev and eval client has 26 files
      models = db.models(groups=g)
      self.assertEqual(len(db.files(groups=g)), 26 * len(models))
      for model_id in models:
        # two enrol files
        self.assertEqual(len(db.files(groups=g, model_ids = model_id, purposes='enrol')), 2)

        # 24 probe files for the (default) 'all' protocol
        self.assertEqual(len(db.files(groups=g, model_ids = model_id, purposes='probe')), 24 * len(models))
        # 6 probe files for the 'expression' protocol 
        self.assertEqual(len(db.files(groups=g, model_ids = model_id, purposes='probe', protocol='expression')), 6 * len(models))
        # 6 probe files for the 'illumination' protocol 
        self.assertEqual(len(db.files(groups=g, model_ids = model_id, purposes='probe', protocol='illumination')), 6 * len(models))
        # 4 probe files for the 'occlusion' protocol 
        self.assertEqual(len(db.files(groups=g, model_ids = model_id, purposes='probe', protocol='occlusion')), 4 * len(models))
        # and finally 8 probe files for the 'illuminatio_and_occlusion' protocol 
        self.assertEqual(len(db.files(groups=g, model_ids = model_id, purposes='probe', protocol='occlusion_and_illumination')), 8 * len(models))
       
    
# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(ARfaceDatabaseTest)
