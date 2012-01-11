#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 13 Jul 18:31:16 2011 

"""All kinds of tests on the DataShuffler class
"""

import os, sys
import unittest
import time
import bob
import numpy

class DataShufferTest(unittest.TestCase):
  """Performs various shuffer tests."""

  def setUp(self):
    
    self.set1 = bob.io.Arrayset()
    self.data1 = numpy.array([1, 0, 0], dtype='float64')
    self.target1 = numpy.array([1], dtype='float64')
    self.set1.append(self.data1)
    self.set1.append(self.data1*2)
    self.set1.append(self.data1*3)

    self.set2 = bob.io.Arrayset()
    self.data2 = numpy.array([0, 1, 0], dtype='float64')
    self.target2 = numpy.array([2], dtype='float64')
    self.set2.append(self.data2)
    self.set2.append(self.data2*2)
    self.set2.append(self.data2*3)

    self.set3 = bob.io.Arrayset()
    self.data3 = numpy.array([0, 0, 1], dtype='float64')
    self.target3 = numpy.array([3], dtype='float64')
    self.set3.append(self.data3)
    self.set3.append(self.data3*2)
    self.set3.append(self.data3*3)
  
  def test01_Initialization(self):

    # Test if we can correctly initialize the shuffler

    shuffle = bob.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])

    self.assertEqual(shuffle.dataWidth, 3)
    self.assertEqual(shuffle.targetWidth, 1)

  def test02_Drawing(self):

    # Tests that drawing works in a particular way

    N = 6 #multiple of number of classes

    shuffle = bob.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])

    [data, target] = shuffle(N)

    self.assertEqual(data.shape, (N, shuffle.dataWidth))
    self.assertEqual(target.shape, (N, shuffle.targetWidth))

    # Finally, we also test if the data is well separated. We have to have 2 
    # of each class since N is multiple of 9
    class1_count = len([data[i,:] for i in range(N) \
        if numpy.dot(data[i,:], self.data1) != 0])
    self.assertEqual(class1_count, 2)
    class2_count = len([data[i,:] for i in range(N) \
        if numpy.dot(data[i,:], self.data2) != 0]) 
    self.assertEqual(class2_count, 2)
    class3_count = len([data[i,:] for i in range(N) \
        if numpy.dot(data[i,:], self.data3) != 0]) 
    self.assertEqual(class3_count, 2)

    N = 28 #not multiple anymore

    [data, target] = shuffle(N)

    self.assertEqual(data.shape, (N, shuffle.dataWidth))
    self.assertEqual(target.shape, (N, shuffle.targetWidth))

    # Finally, we also test if the data is well separated. We have to have 2 
    # of each class since N is multiple of 9
    class1_count = len([data[i,:] for i in range(N) \
        if numpy.dot(data[i,:], self.data1) != 0]) 
    self.assertEqual(class1_count, 10)
    class2_count = len([data[i,:] for i in range(N) \
        if numpy.dot(data[i,:], self.data2) != 0]) 
    self.assertEqual(class2_count, 9)
    class3_count = len([data[i,:] for i in range(N) \
        if numpy.dot(data[i,:], self.data3) != 0]) 
    self.assertEqual(class3_count, 9)

  def test03_Seeding(self):

    # Test if we can correctly set the seed and that this act is effective

    # First test that, by making two shufflers, we get different replies
    shuffle1 = bob.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])
    shuffle2 = bob.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])

    N = 100

    [data1, target1] = shuffle1(N)
    [data2, target2] = shuffle2(N)

    self.assertFalse( (data1 == data2).all() )
    # Note targets will always be the same given N because of the internal
    # design of the C++ DataShuffler.

    # Now show that by drawing twice does not get the same replies!
    # This indicates that the internal random generator is updated at each draw
    # as one expects.
    [data1_2, target1_2] = shuffle1(N)
    
    self.assertFalse( (data1 == data1_2).all() )

    # Finally show that, by setting the seed, we can get the same results
    shuffle1 = bob.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])
    shuffle2 = bob.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])

    # A great seed if you are working in python (the microseconds)
    rng1 = bob.core.random.mt19937(32)
    rng2 = bob.core.random.mt19937(32)

    [data1, target1] = shuffle1(rng1, N)
    [data2, target2] = shuffle2(rng2, N)

    self.assertTrue( (data1 == data2).all() )

  def test04_Normalization(self):

    # Tests that the shuffler can get the std. normalization right
    # Compares results to numpy
    shuffle = bob.trainer.DataShuffler([self.set1, self.set2, self.set3],
        [self.target1, self.target2, self.target3])
  
    npy = numpy.array([[1,0,0], [2,0,0], [3,0,0], 
      [0,1,0], [0,2,0], [0,3,0],
      [0,0,1], [0,0,2], [0,0,3]], 'float64')
    precalc_mean = numpy.array(numpy.mean(npy,0))
    precalc_stddev = numpy.array(numpy.std(npy,0, ddof=1))
    [mean, stddev] = shuffle.stdnorm()

    self.assertTrue( (mean == precalc_mean).all() )
    self.assertTrue( (stddev == precalc_stddev).all() )

    # Now we set the stdnorm flag on and expect data
    self.assertFalse( shuffle.auto_stdnorm )
    shuffle.auto_stdnorm = True
    self.assertTrue( shuffle.auto_stdnorm )

    [data, target] = shuffle(10000)

    # Makes sure the data is approximately zero mean and has std.dev. ~ 1
    # Note: Results will not be of a better precision because we only have 9
    # samples in the Shuffler...
    self.assertEqual( round(data.mean()), 0 )
    self.assertEqual( round(numpy.std(data, ddof=1)), 1 )

  def test05_NormalizationBig(self):

    rng = bob.core.random.mt19937()

    set1 = bob.io.Arrayset()
    draw25 = bob.core.random.normal_float64(mean=2.0, sigma=5.0)
    for i in range(10000):
      set1.append(numpy.array([draw25(rng)], dtype='float64'))
    target1 = numpy.array([1], dtype='float64')

    set2 = bob.io.Arrayset()
    draw32 = bob.core.random.normal_float64(mean=3.0, sigma=2.0)
    for i in range(10000):
      set2.append(numpy.array([draw32(rng)], dtype='float64'))
    target2 = numpy.array([2], dtype='float64')

    shuffle = bob.trainer.DataShuffler([set1, set2], [target1, target2])
    shuffle.auto_stdnorm = True
    prev_mean, prev_stddev = shuffle.stdnorm()

    [data, target] = shuffle(200000)
    self.assertTrue( abs(data.mean()) < 1e-1 )
    self.assertTrue( abs(numpy.std(data, ddof=1) - 1.0) < 1e-1 )

    #note that resetting auto_stdnorm will make the whole go back to normal,
    #but the std normalization values remain the same...
    shuffle.auto_stdnorm = False
    back_mean, back_stddev = shuffle.stdnorm()
    self.assertTrue( abs( (back_mean   - prev_mean  ).sum() ) < 1e-10)
    self.assertTrue( abs( (back_stddev - prev_stddev).sum() ) < 1e-10)


if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStart'):
    bob.core.ProfilerStart(os.environ['BOB_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('BOB_PROFILE') and \
      os.environ['BOB_PROFILE'] and \
      hasattr(bob.core, 'ProfilerStop'):
    bob.core.ProfilerStop()

