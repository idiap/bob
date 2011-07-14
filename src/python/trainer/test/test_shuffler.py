#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 13 Jul 18:31:16 2011 

"""All kinds of tests on the DataShuffler class
"""

import os, sys
import unittest
import time
import torch

class DataShufferTest(unittest.TestCase):
  """Performs various shuffer tests."""

  def setUp(self):
    
    self.set1 = torch.io.Arrayset()
    self.data1 = torch.core.array.array([1, 0, 0], dtype='float64')
    self.target1 = torch.core.array.array([1], dtype='float64')
    self.set1.append(self.data1)
    self.set1.append(self.data1*2)
    self.set1.append(self.data1*3)

    self.set2 = torch.io.Arrayset()
    self.data2 = torch.core.array.array([0, 1, 0], dtype='float64')
    self.target2 = torch.core.array.array([2], dtype='float64')
    self.set2.append(self.data2)
    self.set2.append(self.data2*2)
    self.set2.append(self.data2*3)

    self.set3 = torch.io.Arrayset()
    self.data3 = torch.core.array.array([0, 0, 1], dtype='float64')
    self.target3 = torch.core.array.array([3], dtype='float64')
    self.set3.append(self.data3)
    self.set3.append(self.data3*2)
    self.set3.append(self.data3*3)
  
  def test01_Initialization(self):

    # Test if we can correctly initialize the shuffler

    shuffle = torch.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])

    self.assertEqual(shuffle.dataWidth, 3)
    self.assertEqual(shuffle.targetWidth, 1)

  def test02_Drawing(self):

    # Tests that drawing works in a particular way

    N = 6 #multiple of number of classes

    shuffle = torch.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])

    [data, target] = shuffle(N)

    self.assertEqual(data.shape(), (N, shuffle.dataWidth))
    self.assertEqual(target.shape(), (N, shuffle.targetWidth))

    # Finally, we also test if the data is well separated. We have to have 2 
    # of each class since N is multiple of 9
    class1_count = len([data[i,:] for i in range(N) \
        if torch.math.dot_(data[i,:], self.data1) != 0]) 
    self.assertEqual(class1_count, 2)
    class2_count = len([data[i,:] for i in range(N) \
        if torch.math.dot_(data[i,:], self.data2) != 0]) 
    self.assertEqual(class2_count, 2)
    class3_count = len([data[i,:] for i in range(N) \
        if torch.math.dot_(data[i,:], self.data3) != 0]) 
    self.assertEqual(class3_count, 2)

    N = 28 #not multiple anymore

    [data, target] = shuffle(N)

    self.assertEqual(data.shape(), (N, shuffle.dataWidth))
    self.assertEqual(target.shape(), (N, shuffle.targetWidth))

    # Finally, we also test if the data is well separated. We have to have 2 
    # of each class since N is multiple of 9
    class1_count = len([data[i,:] for i in range(N) \
        if torch.math.dot_(data[i,:], self.data1) != 0]) 
    self.assertEqual(class1_count, 10)
    class2_count = len([data[i,:] for i in range(N) \
        if torch.math.dot_(data[i,:], self.data2) != 0]) 
    self.assertEqual(class2_count, 9)
    class3_count = len([data[i,:] for i in range(N) \
        if torch.math.dot_(data[i,:], self.data3) != 0]) 
    self.assertEqual(class3_count, 9)

  def test03_Seeding(self):

    # Test if we can correctly set the seed and that this act is effective

    # First test that, by making two shufflers, we get the same replies!
    shuffle1 = torch.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])
    shuffle2 = torch.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])

    N = 100

    [data1, target1] = shuffle1(N)
    [data2, target2] = shuffle2(N)

    self.assertTrue( (data1 == data2).all() )
    # Note targets will always be the same given N because of the internal
    # design of the C++ DataShuffler.

    # Now show that by drawing twice does not get the same replies!
    # This indicates that the internal random generator is updated at each draw
    # as one expects.
    [data1_2, target1_2] = shuffle1(N)
    
    self.assertFalse( (data1 == data1_2).all() )

    # Finally show that, by setting the seed, we also can get a different data
    # distribution.
    shuffle1 = torch.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])
    shuffle2 = torch.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])

    # A great seed if you are working in python (the microseconds)
    shuffle1.seed(int(1e12 * (time.time() - int(time.time()))))

    [data1, target1] = shuffle1(N)
    [data2, target2] = shuffle2(N)

    self.assertFalse( (data1 == data2).all() )

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()

