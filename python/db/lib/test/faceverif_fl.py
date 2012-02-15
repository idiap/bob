#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tue Sep 6 12:19:18 2011 +0200
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

"""A few checks for the face verification based on file lists database.
"""

import os, sys
import unittest
import bob

class Faceverif_flDatabaseTest(unittest.TestCase):
  """Performs various tests on the faceverif_fl database."""

  def test01_query(self):

    db = bob.db.faceverif_fl.Database('data/fl')
    self.assertEqual(len(db.models()), 6) # 6 model ids for world, dev and eval
    self.assertEqual(len(db.models(groups='world')), 2) # 2 model ids for world
    self.assertEqual(len(db.models(groups='dev')), 2) # 2 model ids for dev
    self.assertEqual(len(db.models(groups='eval')), 2) # 2 model ids for eval

    self.assertEqual(len(db.Tmodels()), 2) # 2 model ids for T-Norm score normalisation
    self.assertEqual(len(db.Zmodels()), 2) # 2 model ids for Z-Norm score normalisation

    self.assertEqual(len(db.objects(groups='world')), 8) # 8 samples in the world set

    self.assertEqual(len(db.objects(groups='dev', purposes='enrol')), 8) # 8 samples for enrolment in the dev set
    self.assertEqual(len(db.objects(groups='dev', purposes='probe', classes='client')), 8) # 8 samples as client probes in the dev set
    self.assertEqual(len(db.objects(groups='dev', purposes='probe', classes='impostor')), 4) # 4 samples as impostor probes in the dev set

    self.assertEqual(len(db.Tobjects(groups='dev')), 8) # 8 samples for enroling T-norm models
    self.assertEqual(len(db.Zobjects(groups='dev')), 8) # 8 samples for Z-norm impostor accesses

    self.assertEqual(len(db.objects(groups='eval', purposes='enrol')), 8) # 8 samples for enrolment in the dev set
    self.assertEqual(len(db.objects(groups='eval', purposes='probe', classes='client')), 8) # 8 samples as client probes in the dev set
    self.assertEqual(len(db.objects(groups='eval', purposes='probe', classes='impostor')), 0) # 0 samples as impostor probes in the dev set
    
    self.assertEqual(len(db.Tobjects(groups='eval')), 8) # 8 samples for enroling T-norm models
    self.assertEqual(len(db.Zobjects(groups='eval')), 8) # 8 samples for Z-norm impostor accesses

    self.assertEqual(db.getClientIdFromModelId('1'), '1')
    self.assertEqual(db.getClientIdFromModelId('3'), '3')
    self.assertEqual(db.getClientIdFromModelId('6'), '6')
    self.assertEqual(db.getClientIdFromTmodelId('7'), '7')

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(Faceverif_flDatabaseTest)
