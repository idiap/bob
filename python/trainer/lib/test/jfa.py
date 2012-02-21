#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tue Jul 19 12:16:17 2011 +0200
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

"""Test JFA trainer package
"""

import os, sys
import unittest
import bob
import random
import numpy

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class JFATrainerTest(unittest.TestCase):
  """Performs various JFA trainer tests."""
  
  def test01_updateEigen(self):
    # Tests our updateEigen function
    A = numpy.array([0.9293, 0.2511, 0.3517, 0.3500, 0.6160, 0.8308, 0.1966,
      0.4733, 0.5853, 0.5497, 0.7572, 0.5678, 0.9172, 0.7537, 0.0759, 0.2858,
      0.3804, 0.0540]).reshape(2,3,3)

    C = numpy.array([0.1361, 0.5499, 0.6221, 0.4018, 0.1233, 0.4173, 0.9448,
      0.3377, 0.8693, 0.1450, 0.3510, 0.0760, 0.1839, 0.0497, 0.4909, 0.9001,
      0.5797, 0.8530, 0.5132, 0.2399, 0.2400, 0.9027, 0.4893,
      0.3692]).reshape(3,8)

    uv = numpy.ndarray((3,8), 'float64')

    # call the update_u function
    bob.trainer.jfa_updateEigen(A,C,uv)
    # Expected results (JFA cookbook, matlab)
    uv_ref = numpy.array([-0.3120, 0.8754, 0.7009, 0.5404, -0.9020, -5.2395,
      -1.3741, 0.4054, -1.2266, 20.2066, 7.6548, 5.4262, 1.4229, 6.7651,
      2.3354, 0.7859, 2.0873, -15.1771, -5.5488, -4.1597, -0.8070, -3.2141,
      -0.1202, -0.8457]).reshape(3,8)

    # Makes sure results are good
    self.assertTrue(equals(uv, uv_ref, 5e-3))

  def test02_estimateXandU(self):
    # test the estimateXandU function
    F = numpy.array([0.3833, 0.6173, 0.5755, 0.5301, 0.2751, 0.2486, 0.4516,
      0.2277, 0.8044, 0.9861, 0.0300, 0.5357, 0.0871, 0.8021, 0.9891, 0.0669,
      0.9394, 0.0182, 0.6838, 0.7837, 0.5341, 0.8854, 0.8990,
      0.6259]).reshape(4,6)
    N = numpy.array([0.1379, 0.2178, 0.1821, 0.0418, 0.1069, 0.6164, 0.9397,
      0.3545]).reshape(4,2)
    m = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606,
      0.3839]).reshape(6,)
    E = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131]).reshape(6,)
    d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668]).reshape(6,)
    v = numpy.array([0.3367, 0.6624, 0.2442, 0.2955, 0.6802, 0.5278, 0.4116,
      0.6026, 0.7505, 0.5835, 0.5518, 0.5836]).reshape(2,6)
    u = numpy.array([0.5118, 0.0826, 0.7196, 0.9962, 0.3545, 0.9713, 0.3464,
      0.8865, 0.4547, 0.4134, 0.2177, 0.1257]).reshape(2,6)
    z = numpy.array([0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432, 0.9223,
      0.7710, 0.0427, 0.3782, 0.7043, 0.7295]).reshape(2,6)
    y = numpy.array([0.2243, 0.2691, 0.6730, 0.4775]).reshape(2,2)
    x = numpy.ndarray((4,2), 'float64')
    spk_ids = numpy.array([0,0,1,1], 'uint32')

    # call the estimateXandU function
    bob.trainer.jfa_estimateXandU(F,N,m,E,d,v,u,z,y,x,spk_ids)
    # Expected results(JFA cookbook, matlab)
    x_ref = numpy.array([0.2143, 3.1979, 1.8275, 0.1227, -1.3861, 5.3326,
      0.2359,  -0.7914]).reshape(4,2)

    # Makes sure results are good
    self.assertTrue(equals(x, x_ref, 2e-4))

  def test03_estimateYandV(self):
    # test the estimateYandV function
    F = numpy.array(
      [0.3833, 0.6173, 0.5755, 0.5301, 0.2751, 0.2486,
      0.4516, 0.2277, 0.8044, 0.9861, 0.0300, 0.5357,
      0.0871, 0.8021, 0.9891, 0.0669, 0.9394, 0.0182,
      0.6838, 0.7837, 0.5341, 0.8854, 0.8990, 0.6259]).reshape((4,6))
    N = numpy.array(
      [0.1379, 0.2178, 
      0.1821, 0.0418, 
      0.1069, 0.6164, 
      0.9397, 0.3545]).reshape((4,2))
    m = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
    E = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
    d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
    v = numpy.array(
      [0.3367, 0.6624, 0.2442, 0.2955, 0.6802, 0.5278,
      0.4116, 0.6026, 0.7505, 0.5835, 0.5518, 0.5836]).reshape((2,6))
    u = numpy.array(
      [0.5118, 0.0826, 0.7196, 0.9962, 0.3545, 0.9713,
      0.3464, 0.8865, 0.4547, 0.4134, 0.2177, 0.1257]).reshape((2,6))
    z = numpy.array(
      [0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432,
      0.9223, 0.7710, 0.0427, 0.3782, 0.7043, 0.7295]).reshape((2,6))
    y = numpy.ndarray((2,2), 'float64')
    x = numpy.array( [0.9976, 0.1375, 0.8116, 0.3900, 0.4857, 0.9274, 0.8944,
      0.9175]).reshape((4,2))
    spk_ids= numpy.array([0,0,1,1], 'uint32')

    # call the estimateXandU function
    bob.trainer.jfa_estimateYandV(F,N,m,E,d,v,u,z,y,x,spk_ids)

    # Expected results(JFA cookbook, matlab)
    y_ref = numpy.array( [0.9630, 1.3868, 0.04255, -0.3721]).reshape((2,2))

    # Makes sure results are good
    self.assertTrue(equals(y, y_ref, 2e-4))

  def test04_estimateZandD(self):
    # test the estimateYandV function
    F = numpy.array( [0.3833, 0.6173, 0.5755, 0.5301, 0.2751, 0.2486, 0.4516,
      0.2277, 0.8044, 0.9861, 0.0300, 0.5357, 0.0871, 0.8021, 0.9891, 0.0669,
      0.9394, 0.0182, 0.6838, 0.7837, 0.5341, 0.8854, 0.8990, 0.6259]).reshape((4,6))
    N = numpy.array( [0.1379, 0.2178, 0.1821, 0.0418, 0.1069, 0.6164, 0.9397,
      0.3545]).reshape((4,2))
    m = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
    E = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
    d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
    v = numpy.array( [0.3367, 0.6624, 0.2442, 0.2955, 0.6802, 0.5278, 0.4116,
      0.6026, 0.7505, 0.5835, 0.5518, 0.5836]).reshape((2,6))
    u = numpy.array( [0.5118, 0.0826, 0.7196, 0.9962, 0.3545, 0.9713, 0.3464,
      0.8865, 0.4547, 0.4134, 0.2177, 0.1257]).reshape((2,6))
    z = numpy.ndarray((2,6), 'float64')
    y = numpy.array([0.2243, 0.2691, 0.6730, 0.4775]).reshape((2,2))
    x = numpy.array( [0.9976, 0.1375, 0.8116, 0.3900, 0.4857, 0.9274, 0.8944,
      0.9175]).reshape((4,2))
    spk_ids = numpy.array([0,0,1,1], 'uint32')

    # call the estimateXandU function
    bob.trainer.jfa_estimateZandD(F,N,m,E,d,v,u,z,y,x,spk_ids)

    # Expected results(JFA cookbook, matlab)
    z_ref = numpy.array( [0.3256, 1.8633, 0.6480, 0.8085, -0.0432, 0.2885,
      -0.3324, -0.1474, -0.4404, -0.4529, 0.0484, -0.5848]).reshape((2,6))

    # Makes sure results are good
    self.assertTrue(equals(z, z_ref, 2e-4))


  def test05_JFABaseTrainer_updateYandV(self):
    # test the JFABaseTrainer for updating Y and V

    F1 = numpy.array( [0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301,
      0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6,2))
    F2 = numpy.array( [0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669,
      0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6,2))
    F=[F1, F2]

    N1 = numpy.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2,2))
    N2 = numpy.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2,2))
    N=[N1, N2]

    gs11 = bob.machine.GMMStats(2,3)
    gs11.n = N1[:,0]
    gs11.sumPx = F1[:,0].reshape(2,3)
    gs12 = bob.machine.GMMStats(2,3)
    gs12.n = N1[:,1]
    gs12.sumPx = F1[:,1].reshape(2,3)

    gs21 = bob.machine.GMMStats(2,3)
    gs21.n = N2[:,0]
    gs21.sumPx = F2[:,0].reshape(2,3)
    gs22 = bob.machine.GMMStats(2,3)
    gs22.n = N2[:,1]
    gs22.sumPx = F2[:,1].reshape(2,3)

    vec = [[gs11, gs12], [gs21, gs22]]


    m = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
    E = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
    d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
    v = numpy.array( [0.3367, 0.4116, 0.6624, 0.6026, 0.2442, 0.7505, 0.2955,
      0.5835, 0.6802, 0.5518, 0.5278,0.5836]).reshape((6,2))
    v_ref = numpy.array( [0.7228, 0.7892, 0.6475, 0.6080, 0.8631, 0.8416,
      1.6512, 1.6068, 0.0500, 0.0101, 0.4325, 0.6719]).reshape((6,2))
    u = numpy.array( [0.5118, 0.3464, 0.0826, 0.8865, 0.7196, 0.4547, 0.9962,
      0.4134, 0.3545, 0.2177, 0.9713, 0.1257]).reshape((6,2))

    z1 = numpy.array([0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432])
    z2 = numpy.array([0.9223, 0.7710, 0.0427, 0.3782, 0.7043, 0.7295])
    y1 = numpy.array([0., 0.])
    y2 = numpy.array([0., 0.])
    y3 = numpy.array([0.9630, 1.3868])
    y4 = numpy.array([0.0426, -0.3721])
    x1 = numpy.array([0.9976, 0.8116, 0.1375, 0.3900]).reshape((2,2))
    x2 = numpy.array([0.4857, 0.8944, 0.9274, 0.9175]).reshape((2,2))
    z=[z1, z2]
    y=[y1, y2]
    x=[x1, x2]

    # call the updateY function
    ubm = bob.machine.GMMMachine(2,3)
    ubm.meanSupervector = m
    ubm.varianceSupervector = E
    jfam = bob.machine.JFABaseMachine(ubm,2,2)
    jfam.U = u
    jfam.V = v
    jfam.D = d
    jfat = bob.trainer.JFABaseTrainer(jfam)

    jfat.__initNid__(vec)
    jfat.__precomputeSumStatisticsN__(vec)
    jfat.__precomputeSumStatisticsF__(vec)
    jfat.__setSpeakerFactors__(x,y,z)

    jfat.__updateY__(vec)
    jfat.__updateV__(vec)

    # Expected results(JFA cookbook, matlab)
    self.assertTrue(equals(jfat.__Y__[0], y3, 2e-4))
    self.assertTrue(equals(jfat.__Y__[1], y4, 2e-4))
    self.assertTrue(equals(jfam.V, v_ref, 2e-4))


  def test06_JFABaseTrainer_updateXandU(self):
    # test the JFABaseTrainer for updating X and U

    F1 = numpy.array( [0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301,
      0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6,2))
    F2 = numpy.array( [0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669,
      0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6,2))
    F=[F1, F2]

    N1 = numpy.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2,2))
    N2 = numpy.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2,2))
    N=[N1, N2]

    gs11 = bob.machine.GMMStats(2,3)
    gs11.n = N1[:,0]
    gs11.sumPx = F1[:,0].reshape(2,3)
    gs12 = bob.machine.GMMStats(2,3)
    gs12.n = N1[:,1]
    gs12.sumPx = F1[:,1].reshape(2,3)

    gs21 = bob.machine.GMMStats(2,3)
    gs21.n = N2[:,0]
    gs21.sumPx = F2[:,0].reshape(2,3)
    gs22 = bob.machine.GMMStats(2,3)
    gs22.n = N2[:,1]
    gs22.sumPx = F2[:,1].reshape(2,3)

    vec = [[gs11, gs12], [gs21, gs22]]


    m = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
    E = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
    d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
    v = numpy.array( [0.3367, 0.4116, 0.6624, 0.6026, 0.2442, 0.7505, 0.2955,
      0.5835, 0.6802, 0.5518, 0.5278,0.5836]).reshape((6,2))
    u = numpy.array( [0.5118, 0.3464, 0.0826, 0.8865, 0.7196, 0.4547, 0.9962,
      0.4134, 0.3545, 0.2177, 0.9713, 0.1257]).reshape((6,2))

    u_ref = numpy.array( [0.6729, 0.3408, 0.0544, 1.0653, 0.5399, 1.3035,
      2.4995, 0.4385, 0.1292, -0.0576, 1.1962, 0.0117]).reshape((6,2))

    z1 = numpy.array([0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432])
    z2 = numpy.array([0.9223, 0.7710, 0.0427, 0.3782, 0.7043, 0.7295])
    y1 = numpy.array([0.2243, 0.2691])
    y2 = numpy.array([0.6730, 0.4775])
    x1 = numpy.array([0., 0., 0., 0.]).reshape((2,2))
    x2 = numpy.array([0., 0., 0., 0.]).reshape((2,2))
    x3 = numpy.array([0.2143, 1.8275, 3.1979, 0.1227]).reshape((2,2))
    x4 = numpy.array([-1.3861, 0.2359, 5.3326, -0.7914]).reshape((2,2))
    z  = [z1, z2]
    y  = [y1, y2]
    x  = [x1, x2]

    # call the updateX function
    ubm = bob.machine.GMMMachine(2,3)
    ubm.meanSupervector = m
    ubm.varianceSupervector = E
    jfam = bob.machine.JFABaseMachine(ubm,2,2)
    jfam.U = u
    jfam.V = v
    jfam.D = d
    jfat = bob.trainer.JFABaseTrainer(jfam)

    jfat.__initNid__(vec)
    jfat.__precomputeSumStatisticsN__(vec)
    jfat.__precomputeSumStatisticsF__(vec)
    jfat.__setSpeakerFactors__(x,y,z)

    jfat.__updateX__(vec)
    jfat.__updateU__(vec)

    # Expected results(JFA cookbook, matlab)
    self.assertTrue(equals(jfat.__X__[0], x3, 2e-4))
    self.assertTrue(equals(jfat.__X__[1], x4, 2e-4))
    self.assertTrue(equals(jfam.U, u_ref, 2e-4))


  def test07_JFABaseTrainer_updateZandD(self):
    # test the JFABaseTrainer for updating Z and D

    F1 = numpy.array( [0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301,
      0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6,2))
    F2 = numpy.array( [0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669,
      0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6,2))
    F=[F1, F2]

    N1 = numpy.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2,2))
    N2 = numpy.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2,2))
    N=[N1, N2]

    gs11 = bob.machine.GMMStats(2,3)
    gs11.n = N1[:,0]
    gs11.sumPx = F1[:,0].reshape(2,3)
    gs12 = bob.machine.GMMStats(2,3)
    gs12.n = N1[:,1]
    gs12.sumPx = F1[:,1].reshape(2,3)

    gs21 = bob.machine.GMMStats(2,3)
    gs21.n = N2[:,0]
    gs21.sumPx = F2[:,0].reshape(2,3)
    gs22 = bob.machine.GMMStats(2,3)
    gs22.n = N2[:,1]
    gs22.sumPx = F2[:,1].reshape(2,3)

    vec = [[gs11, gs12], [gs21, gs22]]


    m = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
    E = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
    d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
    d_ref = numpy.array([0.3110, 1.0138, 0.8297, 1.0382, 0.0095, 0.6320])
    v = numpy.array( [0.3367, 0.4116, 0.6624, 0.6026, 0.2442, 0.7505, 0.2955,
      0.5835, 0.6802, 0.5518, 0.5278,0.5836]).reshape((6,2))
    u = numpy.array( [0.5118, 0.3464, 0.0826, 0.8865, 0.7196, 0.4547, 0.9962,
      0.4134, 0.3545, 0.2177, 0.9713, 0.1257]).reshape((6,2))

    z1 = numpy.array([0., 0., 0., 0., 0., 0.])
    z2 = numpy.array([0., 0., 0., 0., 0., 0.])
    z3_ref = numpy.array([0.3256, 1.8633, 0.6480, 0.8085, -0.0432, 0.2885])
    z4_ref = numpy.array([-0.3324, -0.1474, -0.4404, -0.4529, 0.0484, -0.5848])
    y1 = numpy.array([0.2243, 0.2691])
    y2 = numpy.array([0.6730, 0.4775])
    x1 = numpy.array([0.9976, 0.8116, 0.1375, 0.3900]).reshape((2,2))
    x2 = numpy.array([0.4857, 0.8944, 0.9274, 0.9175]).reshape((2,2))
    z=[z1, z2]
    y=[y1, y2]
    x=[x1, x2]

    # call the updateZ function
    ubm = bob.machine.GMMMachine(2,3)
    ubm.meanSupervector = m
    ubm.varianceSupervector = E
    jfam = bob.machine.JFABaseMachine(ubm,2,2)
    jfam.U = u
    jfam.V = v
    jfam.D = d
    jfat = bob.trainer.JFABaseTrainer(jfam)

    jfat.__initNid__(vec)
    jfat.__precomputeSumStatisticsN__(vec)
    jfat.__precomputeSumStatisticsF__(vec)
    jfat.__setSpeakerFactors__(x,y,z)

    jfat.__updateZ__(vec)
    jfat.__updateD__(vec)

    # Expected results(JFA cookbook, matlab)
    self.assertTrue(equals(jfat.__Z__[0], z3_ref, 2e-4))
    self.assertTrue(equals(jfat.__Z__[1], z4_ref, 2e-4))
    self.assertTrue(equals(jfam.D, d_ref, 2e-4))


  def test08_JFATrainAndEnrol(self):
    # Train and enrol a JFAMachine

    F1 = numpy.array( [0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301,
      0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6,2))
    F2 = numpy.array( [0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669,
      0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6,2))
    F=[F1, F2]

    N1 = numpy.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2,2))
    N2 = numpy.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2,2))
    N=[N1, N2]

    gs11 = bob.machine.GMMStats(2,3)
    gs11.n = N1[:,0]
    gs11.sumPx = F1[:,0].reshape(2,3)
    gs12 = bob.machine.GMMStats(2,3)
    gs12.n = N1[:,1]
    gs12.sumPx = F1[:,1].reshape(2,3)

    gs21 = bob.machine.GMMStats(2,3)
    gs21.n = N2[:,0]
    gs21.sumPx = F2[:,0].reshape(2,3)
    gs22 = bob.machine.GMMStats(2,3)
    gs22.n = N2[:,1]
    gs22.sumPx = F2[:,1].reshape(2,3)

    vec = [[gs11, gs12], [gs21, gs22]]


    m = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
    E = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
    d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
    v = numpy.array( [0.3367, 0.4116, 0.6624, 0.6026, 0.2442, 0.7505, 0.2955,
      0.5835, 0.6802, 0.5518, 0.5278,0.5836]).reshape((6,2))
    u = numpy.array( [0.5118, 0.3464, 0.0826, 0.8865, 0.7196, 0.4547, 0.9962,
      0.4134, 0.3545, 0.2177, 0.9713, 0.1257]).reshape((6,2))

    # Calls the train function
    ubm = bob.machine.GMMMachine(2,3)
    ubm.meanSupervector = m
    ubm.varianceSupervector = E
    jfam = bob.machine.JFABaseMachine(ubm,2,2)
    jfam.U = u
    jfam.V = v
    jfam.D = d
    jfat = bob.trainer.JFABaseTrainer(jfam)
    jfat.trainNoInit(vec,10)

    v_ref = numpy.array([[0.245364911936476, 0.978133261775424], [0.769646805052223, 0.940070736856596], [0.310779202800089, 1.456332053893072],
          [0.184760934399551, 2.265139705602147], [0.701987784039800, 0.081632150899400], [0.074344030229297, 1.090248340917255]], 'float64')
    u_ref = numpy.array([[0.049424652628448, 0.060480486336896], [0.178104127464007, 1.884873813495153], [1.204011484266777, 2.281351307871720],
          [7.278512126426286, -0.390966087173334], [-0.084424326581145, -0.081725474934414], [4.042143689831097, -0.262576386580701]], 'float64')
    d_ref = numpy.array([9.648467e-18, 2.63720683155e-12, 2.11822157653706e-10, 9.1047243e-17, 1.41163442535567e-10, 3.30581e-19], 'float64')

    eps = 1e-10
    self.assertTrue( numpy.allclose(jfam.V, v_ref, eps) )
    self.assertTrue( numpy.allclose(jfam.U, u_ref, eps) )
    self.assertTrue( numpy.allclose(jfam.D, d_ref, eps) )

    # Calls the enrol function
    jfamachine = bob.machine.JFAMachine(jfam)
    jfatrainer = bob.trainer.JFATrainer(jfamachine,jfat)
  
    Ne = numpy.array([0.1579, 0.9245, 0.1323, 0.2458]).reshape((2,2))
    Fe = numpy.array([0.1579, 0.1925, 0.3242, 0.1234, 0.2354, 0.2734, 0.2514, 0.5874, 0.3345, 0.2463, 0.4789, 0.5236]).reshape((6,2))
    gse1 = bob.machine.GMMStats(2,3)
    gse1.n = Ne[:,0]
    gse1.sumPx = Fe[:,0].reshape(2,3)
    gse2 = bob.machine.GMMStats(2,3)
    gse2.n = Ne[:,1]
    gse2.sumPx = Fe[:,1].reshape(2,3)

    gse = [gse1, gse2]
    jfatrainer.enrol(gse, 5)

    y_ref = numpy.array([0.555991469319657, 0.002773650670010], 'float64')
    z_ref = numpy.array([8.2228e-20, 3.15216909492e-13, -1.48616735364395e-10, 1.0625905e-17, 3.7150503117895e-11, 1.71104e-19], 'float64')
    self.assertTrue( numpy.allclose(jfamachine.y, y_ref, eps) )
    self.assertTrue( numpy.allclose(jfamachine.z, z_ref, eps) )


  def test09_ISVTrainAndEnrol(self):
    # Train and enrol an 'ISVMachine'

    F1 = numpy.array( [0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301,
      0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6,2))
    F2 = numpy.array( [0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669,
      0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6,2))
    F=[F1, F2]

    N1 = numpy.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2,2))
    N2 = numpy.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2,2))
    N=[N1, N2]

    gs11 = bob.machine.GMMStats(2,3)
    gs11.n = N1[:,0]
    gs11.sumPx = F1[:,0].reshape(2,3)
    gs12 = bob.machine.GMMStats(2,3)
    gs12.n = N1[:,1]
    gs12.sumPx = F1[:,1].reshape(2,3)

    gs21 = bob.machine.GMMStats(2,3)
    gs21.n = N2[:,0]
    gs21.sumPx = F2[:,0].reshape(2,3)
    gs22 = bob.machine.GMMStats(2,3)
    gs22.n = N2[:,1]
    gs22.sumPx = F2[:,1].reshape(2,3)

    vec = [[gs11, gs12], [gs21, gs22]]


    m = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
    E = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
    u = numpy.array([0.5118, 0.3464, 0.0826, 0.8865, 0.7196, 0.4547, 0.9962,
      0.4134, 0.3545, 0.2177, 0.9713, 0.1257]).reshape((6,2))

    eps = 1e-10
    d_ref = numpy.array([0.39601136, 0.07348469, 0.47712682, 0.44738127, 0.43179856, 0.45086029], 'float64')
    u_ref = numpy.array([[0.855125642430777, 0.563104284748032], [-0.325497865404680, 1.923598985291687], [0.511575659503837, 1.964288663083095], [9.330165761678115, 1.073623827995043], [0.511099245664012, 0.278551249248978], [5.065578541930268, 0.509565618051587]], 'float64')
    z_ref = numpy.array([-0.079315777443826, 0.092702428248543, -0.342488761656616, -0.059922635809136 , 0.133539981073604, 0.213118695516570], 'float64')

    # Calls the train function
    ubm = bob.machine.GMMMachine(2,3)
    ubm.meanSupervector = m
    ubm.varianceSupervector = E
    jfam = bob.machine.JFABaseMachine(ubm,2)
    jfat = bob.trainer.JFABaseTrainer(jfam)
    jfam.U = u
    jfat.trainISVNoInit(vec, 10, 4)

    self.assertTrue( numpy.allclose(jfam.D, d_ref, eps) )
    self.assertTrue( numpy.allclose(jfam.U, u_ref, eps) )


    # Calls the enrol function
    jfamachine = bob.machine.JFAMachine(jfam)
    jfatrainer = bob.trainer.JFATrainer(jfamachine,jfat)
  
    Ne = numpy.array([0.1579, 0.9245, 0.1323, 0.2458]).reshape((2,2))
    Fe = numpy.array([0.1579, 0.1925, 0.3242, 0.1234, 0.2354, 0.2734, 0.2514, 0.5874, 0.3345, 0.2463, 0.4789, 0.5236]).reshape((6,2))
    gse1 = bob.machine.GMMStats(2,3)
    gse1.n = Ne[:,0]
    gse1.sumPx = Fe[:,0].reshape(2,3)
    gse2 = bob.machine.GMMStats(2,3)
    gse2.n = Ne[:,1]
    gse2.sumPx = Fe[:,1].reshape(2,3)

    gse = [gse1, gse2]
    jfatrainer.enrol(gse, 5)
    self.assertTrue( numpy.allclose(jfamachine.z, z_ref, eps) )

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(JFATrainerTest)
