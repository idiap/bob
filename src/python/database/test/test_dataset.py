#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 24 Jan 2011 11:14:04 CET 

"""Tests and examplify the dataset functionality. Loading and accessing the
data.
"""

import os, sys
import unittest
import torch

INPUT_DATABASE = 'test1.xml'

class DatasetTest(unittest.TestCase):
  """Performs various tests for the Torch::database::Dataset object."""
 
  def test01_CanLoad(self):
    # This method demonstrates how to load an XML from scratch. The only thing
    # you have to do is to pass the relative (or absolute) location of the
    # input file containing the dataset to torch.database.Dataset().
    db = torch.database.Dataset(INPUT_DATABASE)

  def test02_CanBrowseArraysets(self):
    # This example shows how to access the arraysets from a dataset and how to
    # access several of its properties.
    db = torch.database.Dataset(INPUT_DATABASE)
    self.assertEqual(len(db.arraysets), 10)
    arrset_props = (
        {'role': 'pattern1', 'elemtype':torch.database.ArrayType.uint16, 'shape':(3,), 'id':1},
        {'role': 'pattern2', 'elemtype':torch.database.ArrayType.uint16, 'shape':(3,), 'id':2},
        {'role': 'pattern3', 'elemtype':torch.database.ArrayType.int16, 'shape':(3,), 'id':3},
        {'role': 'pattern4', 'elemtype':torch.database.ArrayType.float32, 'shape':(3,), 'id':4},
        {'role': 'pattern5', 'elemtype':torch.database.ArrayType.uint16, 'shape':(2,2), 'id':5},
        {'role': 'pattern6', 'elemtype':torch.database.ArrayType.uint16, 'shape':(2,2,2), 'id':6},
        {'role': 'pattern7', 'elemtype':torch.database.ArrayType.uint16, 'shape':(2,2,2,2), 'id':7},
        {'role': 'pattern8', 'elemtype':torch.database.ArrayType.uint32, 'shape':(3,2,2,2), 'id':8},
        {'role': 'pattern9', 'elemtype':torch.database.ArrayType.uint32, 'shape':(3,2,2,20), 'id':9},
        {'role': 'pattern11', 'elemtype':torch.database.ArrayType.complex64, 'shape':(2,), 'id':11},
        )
    for i, arrset in enumerate(db.arraysets):
      # These are a few properties of the Arrayset in question (pointed by
      # 'arrset'. As you can see, most of the C++ methods are bound in python
      # to look like simple object variables (note you don't need the function
      # call operator ()!). You can find more about methods and properties of
      # arraysets by looking its manual using help(torch.dataset.Arrayset).
      self.assertEqual(arrset.role, arrset_props[i]['role'])
      self.assertEqual(arrset.arrayType, arrset_props[i]['elemtype'])
      self.assertEqual(arrset.shape, arrset_props[i]['shape'])
      self.assertEqual(arrset.id, arrset_props[i]['id'])

    # You can also access the arrayset index like it follows. This is similar
    # to accessing db.arraysets, but it puts the arrayset id's as key of a 
    # dictionary so you can access the dataset contents by them.
    self.assertEqual(type(db.arraysetIndex), dict)
    self.assertEqual(len(db.arraysetIndex), len(db.arraysets))
  
  def test03_CanBrowseArraysTransparently(self):
    # There are 2 sub-classes of arraysets in the database: An inlined version
    # in which each individual array that is part of the arrayset is declared
    # within the database. If you look at the test XML file you will see this
    # is the case, for instance, of arrayset with id=1, 3, 4, 5, 6, 7, etc.
    # The second version of an arrayset defers its contents to an external
    # file. The torch API should treat both types transparently. From the API
    # you can only tell if the arrayset is load or not. When the arrayset is
    # loaded, it means all descriptions of every individual array inside the
    # set has been read into the computer memory. When it is not loaded, first
    # use will trigger loading. This is the case of the second arrayset in the
    # dataset. In these examples we will show how to manipulate both in a 
    # transparent way.
    db = torch.database.Dataset(INPUT_DATABASE)

    # browsing arrayset #11 (10th position) which is complex64, inlined
    # Please note the use of the __getitem__ ([]) operator on datasets. It is
    # possible to either put an integer (for retrieving the arrayset with the
    # given arrayset-id) or a string (for retrieving the relationset with the
    # given relationset-name).
    # IMPORTANT: Please note that while the [] functionality is provided
    # throughout the whole dataset bindings, it is more inefficient than
    # loading all elements at once (for example using one of the *Index
    # properties of every element) and using that. The reason is that the []
    # operator will trigger a map search every time it is used. Your call!
    self.assertEqual(db[11].loaded, True) #should be loaded!
    
    # you can use the len() function to find how many arrays are available
    # inside a given arrayset.
    self.assertEqual(len(db[11]), 2) #contains 2 arrays

    # lets get the arrays and iterate over them
    array_ids = (1, 2)
    for i, k in enumerate(db[11].arrays):
      self.assertEqual(array_ids[i], k.id)

    # Needless to say, you can incarnate torch Array objects (python equivalent
    # of the C++ side blitz::Array<>) as simply. You have two options here. You
    # can either copy or refer to the data stored within the database. Copying
    # will get you an independent copy while referring will make the resulting
    # array point to the dataset allocated data. Changing it *will* have an
    # impact on future dataset queries.

    # Here is a copy example. Note that we can address a particular array using
    # a concatencation of __getitem__() ([]) operators. The first instance
    # works for addressing the Arrayset #9 while the second addresses the Array
    # with array-id = 2.
    bzarray = db[11][2].copy() #copy the second array
    self.assertEqual(bzarray[0], complex(9.,3.))
    self.assertEqual(bzarray[1], complex(5.,7.))

    # Please note that if I modify the array, the database continues unmodified
    bzarray[0] = complex(13.,2.5)
    self.assertEqual(bzarray[0], complex(13.,2.5))
    dbvalue = db[11][2].copy() #re-copy the second array
    self.assertNotEqual(dbvalue[0], complex(13.,2.5))

    # This is not at all what would happen if you decide to refer() instead:
    bzarray = db[11][2].refer() #refers to the second array
    self.assertEqual(bzarray[0], complex(9.,3.))
    self.assertEqual(bzarray[1], complex(5.,7.))
    bzarray[0] = complex(13.,2.5)
    self.assertEqual(bzarray[0], complex(13.,2.5))
    dbvalue = db[11][2].refer() #another pointer / same location
    self.assertEqual(dbvalue[0], complex(13.,2.5)) #!

    # The advantages and disadvantages on using one or the other technique are
    # on speed, memory utilisation and accidental DB modification. You should
    # choose whatever fits you better.

    # Note on casting: You can cast blitz arrays using the technique explained
    # in the array documentation, in case you need it. You can also look at
    # test_array.py for more information and code examples.

    # TODO: External arrayset loading is not yet implemented as of today.
    # TODO: External array loading is also not yet implemented.

  def test05_CanBrowseRelationsets(self):
    # This example shows how to access the relationsets from a dataset and how
    # to access several of its properties.
    db = torch.database.Dataset(INPUT_DATABASE)
    self.assertEqual(len(db.relationsets), 1)
    rs = db['pattern-pattern']
    self.assertEqual(rs.name, 'pattern-pattern')
    self.assertEqual(len(rs.relations), 3)
    self.assertEqual(len(rs.rules), 2)
    self.assertEqual(len(rs.roles), len(rs.rules)) #because of the set design

    # You can also access the relationset index like it follows. This is similar
    # to accessing db.relationsets, but it puts the relationset names's as key 
    # of a dictionary so you can access the dataset contents by them.
    self.assertEqual(type(db.relationsetIndex), dict)
    self.assertEqual(len(db.relationsetIndex), len(db.relationsets))

    # You can also use the [] operator to access rules (give role as parameter)
    # or relations (give relation-id as parameter). The same rule as for
    # Datasets apply to the efficiency of the [] operator compared to
    # retrieving indexes directly using one of the *Index() methods.

  def test06_CanBrowseRelations(self):
    # Relations store clustering information of arraysets and arrays in a 
    # dataset. In this test we examplify how to iterate over relations in a 
    # relationset and how to ultimately obtain data from recognizable groupings.

    # In our example database there is a relationset that establishes a simple 
    # relationship between objects from the arraysets with role "pattern1" and
    # arraysets with role "pattern9". You can get information about the rules
    # of the relationset using the `rule` property. Here is an example:
    db = torch.database.Dataset(INPUT_DATABASE)
    
    #equivalent to db['pattern-pattern']['pattern1'].role!
    self.assertEqual(db.relationsets[0].rules[0].role, "pattern1")
    
    #equivalent to db['pattern-pattern']['pattern9'].role!
    self.assertEqual(db.relationsets[0].rules[1].role, "pattern9")

    #equivalent to db['pattern-pattern']['pattern1'].min!
    self.assertEqual(db.relationsets[0].rules[0].min, 1)

    #equivalent to db['pattern-pattern']['pattern9'].min!
    self.assertEqual(db.relationsets[0].rules[1].min, 1)
    
    #equivalent to db['pattern-pattern']['pattern1'].max!
    self.assertEqual(db.relationsets[0].rules[0].max, 1)
    
    #equivalent to db['pattern-pattern']['pattern9'].max!
    self.assertEqual(db.relationsets[0].rules[1].max, 1)

    # You can browse the relationset rules and relations and then cross-relate
    # member array/arrayset identifiers with those in the dataset to get direct
    # access to the data. Of course, this would be the painful way. We provide
    # a method in every member to allow easier access to the arrays they are 
    # pointing to. You only have to give me the list of arraysets in the
    # dataset and we will do the rest. Here is how to use it. To get into it
    # we first access another python facility that indexes the whole of the 
    # relationset into an easily readable python dictionary. Please read the
    # help message of torch.database.Relationset.index for details.
    index = db.relationsets[0].index
    
    # The keys of 'index' are simply the roles. All (rule) roles should be 
    # there.
    self.assertEqual(sorted(index.keys()), sorted([k.role for k in db.relationsets[0].rules] + ['__id__']))

    # In our test case we only have 2 roles being grouped. As a plus, we also
    # append the relation-id of every relation we have into the special
    # dictionary key '__id__', so we must have 3 keys into our index
    # dictionary.
    self.assertEqual(len(index), 3)

    # The next assertion should be always true. The number of member tuples for
    # each role should be the same, across the relationset. Within each tuple,
    # the number of members may vary, depending on the role. We don't test for
    # that here.
    self.assertEqual(len(index['pattern1']), len(db.relationsets[0].relations))
    self.assertEqual(len(index['pattern1']), len(index['pattern9']))
    self.assertEqual(len(index['pattern1']), len(index['__id__']))

    # This is how to access the arrays using the index. You should keep your
    # dataset object close for that operation, as you are going to need the
    # list of arraysets available:
    bzarrays = [k.copy() for k in index['pattern1'][0][0].arrays(db.arraysets)]
    self.assertEqual(len(bzarrays), 1) #a priori knowledge of the dataset
    self.assertEqual(bzarrays[0], torch.core.array.uint16_1([43, 21, 58], (3,)))

    # Another example: access the 3rd relation, 'pattern9'
    bzarrays = [k.copy() for k in index['pattern9'][2][0].arrays(db.arraysets)]
    self.assertEqual(len(bzarrays), 1) #a priori knowledge of the dataset
    self.assertEqual(bzarrays[0].shape(), (3,2,2,20))
    # ... other checks?

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
