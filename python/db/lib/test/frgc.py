#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date:   Fri May 11 17:20:46 CEST 2012
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

"""Sanity checks for the FRGC database.
"""

import os, sys
import unittest
import bob

class FRGCDatabaseTest(unittest.TestCase):
  """Performs some tests on the GBU database."""

  m_db = bob.db.frgc.Database() 

  def test_clients(self):
    """Tests that the 'clients()' and 'models()' functions return the desired number of elements"""

    # the protocols training, dev, idiap
    protocols = self.m_db.m_protocols

    # client counter
    self.assertEqual(len(self.m_db.clients(groups='world')), 222)
    for protocol in protocols:
      self.assertEqual(len(self.m_db.clients(protocol=protocol)), 535)
      self.assertEqual(len(self.m_db.clients(groups='dev', protocol=protocol)), 466)

    # for different mask types, the client counters of 'enrol' and 'probe' are different
    for protocol in protocols:
      self.assertEqual(len(self.m_db.clients(groups='dev', protocol=protocol, purposes='enrol', mask_type='maskI')), 466)
      self.assertEqual(len(self.m_db.clients(groups='dev', protocol=protocol, purposes='probe', mask_type='maskI')), 450)
      self.assertEqual(len(self.m_db.clients(groups='dev', protocol=protocol, purposes='enrol', mask_type='maskII')), 466)
      self.assertEqual(len(self.m_db.clients(groups='dev', protocol=protocol, purposes='probe', mask_type='maskII')), 461)
      self.assertEqual(len(self.m_db.clients(groups='dev', protocol=protocol, purposes='enrol', mask_type='maskIII')), 370)
      self.assertEqual(len(self.m_db.clients(groups='dev', protocol=protocol, purposes='probe', mask_type='maskIII')), 345)
    
    # check the number of models
    self.assertEqual(len(self.m_db.models(groups='world')), 12776)
    self.assertEqual(len(self.m_db.models(groups='dev', protocol='2.0.1', mask_type='maskI')), 14628)
    self.assertEqual(len(self.m_db.models(groups='dev', protocol='2.0.2', mask_type='maskI')), 3657)
    self.assertEqual(len(self.m_db.models(groups='dev', protocol='2.0.4', mask_type='maskI')), 14628)
    self.assertEqual(len(self.m_db.models(groups='dev', protocol='2.0.1', mask_type='maskII')), 15336)
    self.assertEqual(len(self.m_db.models(groups='dev', protocol='2.0.2', mask_type='maskII')), 3834)
    self.assertEqual(len(self.m_db.models(groups='dev', protocol='2.0.4', mask_type='maskII')), 15336)
    self.assertEqual(len(self.m_db.models(groups='dev', protocol='2.0.1', mask_type='maskIII')), 7572)
    self.assertEqual(len(self.m_db.models(groups='dev', protocol='2.0.2', mask_type='maskIII')), 1893)
    self.assertEqual(len(self.m_db.models(groups='dev', protocol='2.0.4', mask_type='maskIII')), 7572)
      

  def test_files(self):
    """Tests that the 'files()' function returns reasonable output"""
    # The number of files should always be identical to the number of models...
    protocols = self.m_db.m_protocols
    masks = self.m_db.m_mask_types
    
    self.assertEqual(len(self.m_db.files(groups='world')), len(self.m_db.models(groups='world')))
    
    for mask in masks:
      for protocol in ('2.0.1', '2.0.4'):
        self.assertEqual(len(self.m_db.files(groups='dev', protocol=protocol, purposes='enrol', mask_type=mask)), 
                         len(self.m_db.models(groups='dev', protocol=protocol, mask_type=mask)))

      # for protocol 4, there are 4 files for each model
      self.assertEqual(len(self.m_db.files(groups='dev', protocol='2.0.2', purposes='enrol', mask_type=mask)), 
                       len(self.m_db.models(groups='dev', protocol='2.0.2', mask_type=mask)) * 4)
    
  def test_file_ids(self):
    """Tests that the client id's returned by the 'get_client_id_from_file_id()' and 'get_client_id_from_model_id()' functions are correct"""

    # this test might take a while...
    for protocol in self.m_db.m_protocols:
      # extract all models
      for model_id in self.m_db.models(groups='dev', protocol=protocol):
        # get the client id of the model
        client_id = self.m_db.get_client_id_from_model_id(model_id)
        # check that all files with the same model id share the same client id
        for file_id in self.m_db.files(groups='dev', protocol=protocol, model_ids=(model_id,), purposes='enrol'):
          self.assertEqual(self.m_db.get_client_id_from_file_id(file_id), client_id)

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(FRGCDatabaseTest)
