#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 24 Jan 2011 17:30:37 CET 

"""Tests and examplify the dataset functionality. This suite exercises the
dataset writing. 
"""

import os, sys
import unittest
import tempfile
import torch

def get_tempfilename(prefix='torchtest_', suffix='.xml'):
  (fd, name) = tempfile.mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

INPUT_DATABASE = 'test1.xml'

class DatasetWritingTest(unittest.TestCase):
  """Performs various tests for the Torch::database::Dataset object."""
 
  def test01_CanResave(self):
    # This test demonstrates how to re-save the database. It uses the most
    # basic primitives for such operation. load() to load the dataset and
    # save() to save it back.
    db = torch.database.Dataset(INPUT_DATABASE)

    # In python, you can also load from a string containing the XML dataset
    # description using torch.database.loadString("<xml..."). Similarly, you 
    # can get a string representation of the Dataset by touching the Datset.xml
    # property. This will test the cycle a bit:
    tmpname = get_tempfilename()
    db.save(tmpname)
    db2 = torch.database.Dataset(tmpname)
    os.unlink(tmpname)

    # Now test loading from a string
    db3 = torch.database.loadString(db2.xml)

    # TODO: check db == db2 == db3

  def todo_test02_CanCreateFromScratch(self):
    # This test demonstrates how to create a very basic but complete Dataset
    # that contains two Arrayset and a single Relationset.

    # First, we create an emtpy Dataset
    db = torch.database.Dataset()
  
    # Here we create two arrays and Array sets to hold them
    for arrayset_role in ("pattern", "target"):
      aset = torch.database.Arrayset()
      # You can optionally set the shape, arrayType, filename and id. We don't
      # in this simple example. The "shape" and "arrayType" will be set
      # automatically when you first stuff an array into the set. The parameter
      # "filename" will determine if the data for this array will be saved
      # externally. The parameter "id" will set the arrayset-id. If you don't
      # set it, one will be attributed automatically based on the arrayset
      # order within the dataset, starting at 1.
      aset.role = arrayset_role
      db.append(aset)

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  os.chdir('data')
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
