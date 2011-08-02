#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue  2 Aug 13:02:21 2011 

"""Tests some of the array concatenation operations.
"""


import os, sys
import unittest
import torch


class ArrayConcatenationTest(unittest.TestCase):
  """Performs various tests for concatenating array objects."""

  def test01_canConcatenate1D(self):

    a = torch.core.array.float64_1([1.,2.,.3], (3,))
    b = torch.core.array.float64_1([.4,5,-6], (3,))

    c = torch.core.array.cat((a,b), 0) #works

    self.assertEqual( c.shape(), (6,) )
    self.assertEqual( c.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(c[:3]) )
    self.assertTrue( b.numeq(c[3:6]) )

    # can only concatenate over existing dimensions on source arrays
    self.assertRaises(IndexError, torch.core.array.cat, (a,b), 1)

  def test02_canStack1D(self):

    a = torch.core.array.float64_1([1.,2.,.3], (3,))
    b = torch.core.array.float64_1([.4,5,-6], (3,))

    z = torch.core.array.stack((a,b))

    self.assertEqual( z.shape(), (2,3) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[0,:]) )
    self.assertTrue( b.numeq(z[1,:]) )

    # note you can only stack arrays with the same shape
    c = torch.core.array.float64_1([.4,5,-6,.7], (4,))

    self.assertRaises(RuntimeError, torch.core.array.stack, (a,c))

  def test03_canConcatenate2D(self):

    a = torch.core.array.float64_2(range(4), (2,2))
    b = a*-2
    c = torch.core.array.float64_2(range(8), (2,4))

    z = torch.core.array.cat((a,b), 0) #works
    self.assertEqual( z.shape(), (4,2) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[:2,:]) )
    self.assertTrue( b.numeq(z[2:,:]) )

    # can only concatenate over existing dimensions on source arrays
    self.assertRaises(IndexError, torch.core.array.cat, (a,b), 2)

    # concatenation will work along any dimension as long as source
    # arrays are compatible.
    z = torch.core.array.cat((a,c), 1)
    self.assertEqual( z.shape(), (2,6) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[:,:2]) )
    self.assertTrue( c.numeq(z[:,2:]) )

    self.assertRaises(RuntimeError, torch.core.array.cat, (a,c), 0)

  def test04_canStack2D(self):

    a = torch.core.array.float64_2(range(4), (2,2))
    b = a*-2
    c = torch.core.array.float64_2(range(8), (2,4))

    z = torch.core.array.stack((a,b))

    self.assertEqual( z.shape(), (2,2,2) )
    self.assertEqual( c.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[0,:,:]) )
    self.assertTrue( b.numeq(z[1,:,:]) )

    # statcking will only work if the array shapes are compatible
    self.assertRaises(RuntimeError, torch.core.array.stack, (a,c))

  def test05_canConcatenate3D(self):

    a = torch.core.array.float64_3(range(8), (2,2,2))
    b = a*-2
    c = torch.core.array.float64_3(range(16), (2,4,2))

    z = torch.core.array.cat((a,b), 0) #works
    self.assertEqual( z.shape(), (4,2,2) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[:2,:,:]) )
    self.assertTrue( b.numeq(z[2:,:,:]) )

    # can only concatenate over existing dimensions on source arrays
    self.assertRaises(IndexError, torch.core.array.cat, (a,b), 3)

    # concatenation will work along any dimension as long as source
    # arrays are compatible.
    z = torch.core.array.cat((a,c), 1)
    self.assertEqual( z.shape(), (2,6,2) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[:,:2,:]) )
    self.assertTrue( c.numeq(z[:,2:,:]) )

    z = torch.core.array.cat((a,b), 2)
    self.assertEqual( z.shape(), (2,2,4) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[:,:,:2]) )
    self.assertTrue( b.numeq(z[:,:,2:]) )

    self.assertRaises(RuntimeError, torch.core.array.cat, (a,c), 0)

  def test06_canStack3D(self):

    a = torch.core.array.float64_3(range(8), (2,2,2))
    b = a*-2
    c = torch.core.array.float64_3(range(16), (2,4,2))

    z = torch.core.array.stack((a,b))

    self.assertEqual( z.shape(), (2,2,2,2) )
    self.assertEqual( c.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[0,:,:,:]) )
    self.assertTrue( b.numeq(z[1,:,:,:]) )

    # statcking will only work if the array shapes are compatible
    self.assertRaises(RuntimeError, torch.core.array.stack, (a,c))

  def test07_canConcatenate4D(self):

    a = torch.core.array.float64_4(range(16), (2,2,2,2))
    b = a*-2
    c = torch.core.array.float64_4(range(32), (2,4,2,2))

    z = torch.core.array.cat((a,b), 0) #works
    self.assertEqual( z.shape(), (4,2,2,2) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[:2,:,:,:]) )
    self.assertTrue( b.numeq(z[2:,:,:,:]) )

    # can only concatenate over existing dimensions on source arrays
    self.assertRaises(IndexError, torch.core.array.cat, (a,b), 4)

    # concatenation will work along any dimension as long as source
    # arrays are compatible.
    z = torch.core.array.cat((a,c), 1)
    self.assertEqual( z.shape(), (2,6,2,2) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[:,:2,:,:]) )
    self.assertTrue( c.numeq(z[:,2:,:,:]) )

    z = torch.core.array.cat((a,b), 2)
    self.assertEqual( z.shape(), (2,2,4,2) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[:,:,:2,:]) )
    self.assertTrue( b.numeq(z[:,:,2:,:]) )

    z = torch.core.array.cat((a,b), 3)
    self.assertEqual( z.shape(), (2,2,2,4) )
    self.assertEqual( z.cxx_element_typename, 'float64' )
    self.assertTrue( a.numeq(z[:,:,:,:2]) )
    self.assertTrue( b.numeq(z[:,:,:,2:]) )

    self.assertRaises(RuntimeError, torch.core.array.cat, (a,c), 0)

if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  #os.chdir(os.path.join('data', 'video'))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
