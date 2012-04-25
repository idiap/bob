#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date: Fri Apr 20 12:04:44 CEST 2012
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

"""Checks for the AT&T database.
"""

import os, sys
import unittest
import bob

class ATNTDatabaseTest(unittest.TestCase):
  """Performs some tests on the AT&T database."""


  def test_query(self):
    db = bob.db.atnt.Database()
    
    f = db.files()
    self.assertEqual(len(f.values()), 400) # number of all files in the database

    f = db.files(groups='train')
    self.assertEqual(len(f.values()), 200) # number of all training files
   
    f = db.files(groups='test')
    self.assertEqual(len(f.values()), 200) # number of all test files
    
    f = db.files(groups='test', purposes = 'enrol')
    self.assertEqual(len(f.values()), 100) # number of enrol files

    f = db.files(groups='test', purposes = 'probe')
    self.assertEqual(len(f.values()), 100) # number of probe files

    f = db.client_ids()
    self.assertEqual(len(f), 40) # number of clients
    
    f = db.client_ids(groups = 'train')
    self.assertEqual(len(f), 20) # number of training clients

    f = db.client_ids(groups = 'test')
    self.assertEqual(len(f), 20) # number of test clients

    f = db.files(groups = 'test', purposes = 'enrol', client_ids = [3])
    self.assertEqual(len(f), 5)
    keys = sorted(f.keys())
    values = sorted(list(db.m_enrol_files))
    for i in range(5):
      self.assertEqual(f[keys[i]], os.path.join("s3", str(values[i])))
      self.assertEqual(db.get_client_id_from_file_id(keys[i]), 3)
    
# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(ATNTDatabaseTest)
