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

    # TODO: check db == db2 == db3

  def test02_CanResaveRelative(self):
    # This test demonstrates how to re-save the database. It uses the most
    # basic primitives for such operation. load() to load the dataset and
    # save() to save it back.
    db = torch.database.Dataset(INPUT_DATABASE)

    # In python, you can also load from a string containing the XML dataset
    # description using torch.database.loadString("<xml..."). Similarly, you 
    # can get a string representation of the Dataset by touching the Datset.xml
    # property. This will test the cycle a bit:
    tmpname = get_tempfilename()
    os.chdir( os.path.dirname(tmpname) )
    db.save( os.path.basename(tmpname) )
    db2 = torch.database.Dataset(tmpname)
    os.unlink(tmpname)

  def test03_CanCreateFromScratch(self):
    # This test demonstrates how to create a very basic but complete Dataset
    # that contains two Arraysets and a single Relationset.

    # First, we create an emtpy Dataset
    db = torch.database.Dataset("Scratch Database from Python", 1)
  
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

    # Now we stuff one complex blitz Array in the "pattern" arrayset and one
    # boolean blitz Array in the "target" arrayset
    db[1].append(torch.core.array.complex128_1(range(4),(4,)))
    db[2].append(torch.core.array.bool_1((True,),(1,)))

    # We now save this dataset
    tmpname = get_tempfilename()
    db.save(tmpname)

    # And re-load to make sure it is what I think it is
    dbr = torch.database.Dataset(tmpname)
    self.assertEqual(db, dbr)

    # We now append a single relationset with a single relation binding the
    # pattern and target arrays in one, we verify everything is working as
    # expected. Relationsets work a little bit differently than Arrays in the
    # sense an automatic name cannot be assigned on append(), so we provide
    # only the classical python set methods through subscripts. Please note you
    # can also use the same technique with arraysets like bellow:
    # db[35] = torch.database.Arrayset()
    db["p-t"] = torch.database.Relationset()

    # We stuff some rules into the Relationset. For this example, every pattern
    # must correspond to a single target, so:
    db["p-t"]["pattern"] = torch.database.Rule(min=1, max=1)
    db["p-t"]["target"] = torch.database.Rule(min=1, max=1)

    # And now we insert our first and only relation. 
    relation = torch.database.Relation()
    relation.add(1, 1) #arrayset.id = 1, array.id = 1
    relation.add(2, 1) #arrayset.id = 2, array.id = 1
    db["p-t"].append(relation) 
    
    # Note that if that does not correspond to the rules, we will raise!

    # Let's create an invalid relation and try to stuff it see what happens:
    relation = torch.database.Relation()
    relation.add(1, 1) #arrayset.id = 1, array.id = 1
    # and suppose I forget to add the second pair:
    self.assertRaises(torch.database.InvalidRelation, db["p-t"].append, relation)

    # Now we save that, overwriting the previous db
    db.save(tmpname)

    # We reload and compare:
    dbr = torch.database.Dataset(tmpname)
    self.assertEqual(db, dbr)

    # And remove the temporary file
    os.unlink(tmpname)

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
