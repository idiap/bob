#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon Aug 8 12:40:24 2011 +0200
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

"""A few checks at the replay attack database.
"""

import os, sys
import unittest
import bob

class ReplayDatabaseTest(unittest.TestCase):
  """Performs various tests on the replay attack database."""

  def test01_queryRealAccesses(self):

    db = bob.db.replay.Database()
    f = db.files(cls='real')
    self.assertEqual(len(set(f.values())), 200) #200 unique auth sessions
    for k,v in f.items():
      self.assertTrue( (v.find('authenticate') != -1) )
      self.assertTrue( (v.find('rcd') != -1) )
      self.assertTrue( (v.find('webcam') != -1) )
    
    train = db.files(cls='real', groups='train')
    self.assertEqual(len(set(train.values())), 60)

    dev = db.files(cls='real', groups='devel')
    self.assertEqual(len(set(dev.values())), 60)

    test = db.files(cls='real', groups='test')
    self.assertEqual(len(set(test.values())), 80)

    #tests train, devel and test files are distinct
    s = set(train.values() + dev.values() + test.values())
    self.assertEqual(len(s), 200)

  def queryAttackType(self, protocol, N):

    db = bob.db.replay.Database()
    f = db.files(cls='attack', protocol=protocol)

    self.assertEqual(len(set(f.values())), N) 
    for k,v in f.items():
      self.assertTrue(v.find('attack') != -1)

    train = db.files(cls='attack', groups='train', protocol=protocol)
    self.assertEqual(len(set(train.values())), int(round(0.3*N)))

    dev = db.files(cls='attack', groups='devel', protocol=protocol)
    self.assertEqual(len(set(dev.values())), int(round(0.3*N)))

    test = db.files(cls='attack', groups='test', protocol=protocol)
    self.assertEqual(len(set(test.values())), int(round(0.4*N)))

    #tests train, devel and test files are distinct
    s = set(train.values() + dev.values() + test.values())
    self.assertEqual(len(s), N)

  def test02_queryAttacks(self):

    self.queryAttackType('grandtest', 1000)
  
  def test03_queryPrintAttacks(self):

    self.queryAttackType('print', 200)
  
  def test04_queryMobileAttacks(self):

    self.queryAttackType('mobile', 400)
  
  def test05_queryHighDefAttacks(self):

    self.queryAttackType('highdef', 400)
  
  def test06_queryPhotoAttacks(self):

    self.queryAttackType('photo', 600)
  
  def test07_queryVideoAttacks(self):

    self.queryAttackType('video', 400)
  
  def test08_queryEnrollments(self):

    db = bob.db.replay.Database()
    f = db.files(cls='enroll')
    self.assertEqual(len(set(f.values())), 100) #50 clients, 2 conditions
    for k,v in f.items():
      self.assertTrue(v.find('enroll') != -1)

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(ReplayDatabaseTest)
