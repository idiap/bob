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

"""Checks for the FIR database.
"""

import os, sys
import unittest
import bob

class FIRDatabaseTest(unittest.TestCase):
  """Performs some tests on the FIR database."""


  def test_query(self):
    db = bob.db.fir.Database()
    
    f = db.files()
    self.assertEqual(len(f.values()), 1463) # number of all files in the database

    f = db.files(groups='world')
    self.assertEqual(len(f.values()), 625) # number of all training files
   
    f = db.files(groups='dev')
    self.assertEqual(len(f.values()), 419) # number of all test files
    
    f = db.files(groups='dev', purposes = 'enrol')
    self.assertEqual(len(f.values()), 120) # number of enrol files

    f = db.files(groups='dev', purposes = 'probe')
    self.assertEqual(len(f.values()), 298) # number of probe files

    f = db.files(groups='eval')
    self.assertEqual(len(f.values()), 418) # number of all test files
    
    f = db.files(groups='eval', purposes = 'enrol')
    self.assertEqual(len(f.values()), 120) # number of enrol files

    f = db.files(groups='eval', purposes = 'probe')
    self.assertEqual(len(f.values()), 298) # number of probe files
    f = db.clients()
    self.assertEqual(len(f), 21) # number of clients
    
    f = db.clients(groups = 'world')
    self.assertEqual(len(f), 12) # number of training clients

    f = db.clients(groups = 'dev')
    self.assertEqual(len(f), 6) # number of test clients

    f = db.clients(groups = 'eval')
    self.assertEqual(len(f), 6) # number of test clients

    f = db.files(groups = 'dev', purposes = 'enrol', client_ids = [2])
    self.assertEqual(len(f), 19)
    
# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(FIRDatabaseTest)
