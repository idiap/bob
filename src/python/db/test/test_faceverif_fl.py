#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 08 Aug 2011 11:10:29 CEST 

"""A few checks for the face verification based on file lists database.
"""

import os, sys
import unittest
import torch

class Faceverif_flDatabaseTest(unittest.TestCase):
  """Performs various tests on the faceverif_fl database."""

  def test01_query(self):

    db = torch.db.faceverif_fl.Database('data/fl')
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

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  #os.chdir('data')
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
