#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <francois.moulin@idiap.ch>

"""Tests machine package
"""

import os, sys
import unittest
import numpy
import bob
import itertools

def my_strip(str):
  return str.strip('\n')

def list_of_lines(lines):
  vals = []
  for line in lines:
    vals.append(map(float, line.rsplit()))
      
  return vals

def read_lightsaber(filename):

  f = open(filename, 'r')
  v = f.readlines()
  f.close()
  v = map(my_strip, v)

  weis  = [v[i] for i in range(0, len(v), 3)]
  means = [v[i] for i in range(1, len(v), 3)]
  varis = [v[i] for i in range(2, len(v), 3)]

  weis  = map(float, weis)
  means = list_of_lines(means)
  varis = list_of_lines(varis)

  return weis, means, varis

class MachineTest(unittest.TestCase):
  """Performs various machine tests."""

  def test01_GMMMachine(self):
    """Test a GMMMachine"""

    sampler = bob.trainer.SimpleFrameSampler(bob.io.Arrayset("data/1028_m_g1_s03_1028_en_2.bindata"))
    gmm = bob.machine.GMMMachine(3, 91)

    weis, means, varis = read_lightsaber("data/tan-triggs-64x80-normalised-D91-gmm512-B24-BANCA.data.s")
    print means

    gmm.weights   = numpy.array(weis, 'float64')
    gmm.means     = numpy.array(means, 'float64')
    gmm.variances = numpy.array(varis, 'float64')

    gmm.print_()

    gmm.forward(sampler)

    stats = bob.machine.GMMStats(3, 91)
    gmm.accStatistics(sampler, stats)
    stats.print_()

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
