#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri Oct 14 18:07:56 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

"""Tests PLDA trainer
"""

import os, sys
import unittest
import bob
import random
import numpy

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class PLDATrainerTest(unittest.TestCase):
  """Performs various PLDA trainer tests."""
  
  def test01_plda_EM(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    D = 7
    nf = 2
    ng = 3

    # first identity (4 samples)
    a = numpy.array([
      [1,2,3,4,5,6,7],
      [7,8,3,3,1,8,2],
      [3,2,1,4,5,1,7],
      [9,0,3,2,1,4,6],
      ], dtype='float64')

    # second identity (3 samples)
    b = numpy.array([
      [5,6,3,4,2,0,2],
      [1,7,8,9,4,4,8],
      [8,7,2,5,1,1,1],
      ], dtype='float64')

    # list of arrays (training data)
    l = [a,b]

    # initial values for F, G and sigma
    G_init=numpy.array([-1.1424, -0.5044, -0.1917,
      -0.6249,  0.1021, -0.8658,
      -1.1687,  1.1963,  0.1807,
      0.3926,  0.1203,  1.2665,
      1.3018, -1.0368, -0.2512,
      -0.5936, -0.8571, -0.2046,
      0.4364, -0.1699, -2.2015]).reshape(D,ng)

    # F <-> PCA on G
    F_init=numpy.array([-0.054222647972093, -0.000000000783146, 
      0.596449127693018,  0.000000006265167, 
      0.298224563846509,  0.000000003132583, 
      0.447336845769764,  0.000000009397750, 
      -0.108445295944185, -0.000000001566292, 
      -0.501559493741856, -0.000000006265167, 
      -0.298224563846509, -0.000000003132583]).reshape(D,nf)
    sigma_init = 0.01 * numpy.ones(D, 'float64')

    # Defines reference results based on Princes'matlab implementation
    # After 1 iteration
    z_first_order_a_1 = numpy.array(
      [-2.624115900658397, -0.000000034277848,  1.554823055585319,  0.627476234024656, -0.264705934182394,
       -2.624115900658397, -0.000000034277848, -2.703482671599357, -1.533283607433197,  0.553725774828231,
       -2.624115900658397, -0.000000034277848,  2.311647528461115,  1.266362142140170, -0.317378177105131,
       -2.624115900658397, -0.000000034277848, -1.163402640008200, -0.372604542926019,  0.025152800097991
      ]).reshape(4, nf+ng)
    z_first_order_b_1 = numpy.array(
      [ 3.494168818797438,  0.000000045643026,  0.111295550530958, -0.029241422535725,  0.257045446451067,
        3.494168818797438,  0.000000045643026,  1.102110715965762,  1.481232954001794, -0.970661225144399,
        3.494168818797438,  0.000000045643026, -1.212854031699468, -1.435946529317718,  0.717884143973377
      ]).reshape(3, nf+ng)
  
    z_second_order_sum_1 = numpy.array(
      [64.203518285366087,  0.000000747228248,  0.002703277337642,  0.078542842475345,  0.020894328259862,
        0.000000747228248,  6.999999999999980, -0.000000003955962,  0.000000002017232, -0.000000003741593,
        0.002703277337642, -0.000000003955962, 19.136889380923918, 11.860493771107487, -4.584339465366988,
        0.078542842475345,  0.000000002017232, 11.860493771107487,  8.771502339750128, -3.905706024997424,
        0.020894328259862, -0.000000003741593, -4.584339465366988, -3.905706024997424,  2.011924970338584
      ]).reshape(nf+ng, nf+ng)

    sigma_1 = numpy.array(
        [2.193659969999207, 3.748361365521041, 0.237835235737085,
          0.558546035892629, 0.209272700958400, 1.717782807724451,
          0.248414618308223])

    F_1 = numpy.array(
        [-0.059083416465692,  0.000000000751007,
          0.600133217253169,  0.000000006957266,
          0.302789123922871,  0.000000000218947,
          0.454540641429714,  0.000000003342540,
          -0.106608957780613, -0.000000001641389,
          -0.494267694269430, -0.000000011059552,
          -0.295956102084270, -0.000000006718366]).reshape(D,nf)

    G_1 = numpy.array(
        [-1.836166150865047,  2.491475145758734,  5.095958946372235,
          -0.608732205531767, -0.618128420353493, -1.085423135463635,
          -0.697390472635929, -1.047900122276840, -6.080211153116984,
          0.769509301515319, -2.763610156675313, -5.972172587527176,
          1.332474692714491, -1.368103875407414, -2.096382536513033,
          0.304135903830416, -5.168096082564016, -9.604769461465978,
          0.597445549865284, -1.347101803379971, -5.900246013340080]).reshape(D,ng)

    # After 2 iterations
    z_first_order_a_2 = numpy.array(
        [-2.144344161196005, -0.000000027851878,  1.217776189037369,  0.232492571855061, -0.212892893868819,
          -2.144344161196005, -0.000000027851878, -2.382647766948079, -1.759951013670071,  0.587213207926731,
          -2.144344161196005, -0.000000027851878,  2.143294830538722,  0.909307594408923, -0.183752098508072,
          -2.144344161196005, -0.000000027851878, -0.662558006326892,  0.717992497547010, -0.202897892977004
      ]).reshape(4, nf+ng)
    z_first_order_b_2 = numpy.array(
        [ 2.695117129662246,  0.000000035005543, -0.156173294945791, -0.123083763746364,  0.271123341933619,
          2.695117129662246,  0.000000035005543,  0.690321563509753,  0.944473716646212, -0.850835940962492,
          2.695117129662246,  0.000000035005543, -0.930970138998433, -0.949736472690315,  0.594216348861889
      ]).reshape(3, nf+ng)
 
    z_second_order_sum_2 = numpy.array(
        [41.602421167226410,  0.000000449434708, -1.513391506933811, -0.477818674270533,  0.059260102368316,
          0.000000449434708,  7.000000000000005, -0.000000023255959, -0.000000005157439, -0.000000003230262,
          -1.513391506933810, -0.000000023255959, 14.399631061987494,  8.068678077509025, -3.227586434905497,
          -0.477818674270533, -0.000000005157439,  8.068678077509025,  7.263248678863863, -3.060665688064639,
          0.059260102368316, -0.000000003230262, -3.227586434905497, -3.060665688064639,  1.705174220723198
      ]).reshape(nf+ng, nf+ng)

    sigma_2 = numpy.array(
      [1.120493935052524, 1.777598857891599, 0.197579528599150,
        0.407657093211478, 0.166216300651473, 1.044336960403809,
        0.287856936559308])
 
    F_2 = numpy.array(
      [-0.111956311978966,  0.000000000781025,
        0.702502767389263,  0.000000007683917,
        0.337823622542517,  0.000000000637302,
        0.551363737526339,  0.000000004854293,
       -0.096561040511417, -0.000000001716011,
       -0.661587484803602, -0.000000012394362,
       -0.346593051621620, -0.000000007134046]).reshape(D,nf)

    G_2 = numpy.array(
      [-2.266404374274820,  4.089199685832099,  7.023039382876370,
        0.094887459097613, -3.226829318470136, -3.452279917194724,
       -0.498398131733141, -1.651712333649899, -6.548008210704172,
        0.574932298590327, -2.198978667003715, -5.131253543126156,
        1.415857426810629, -1.627795701160212, -2.509013676007012,
       -0.543552834305580, -3.215063993186718, -7.006305082499653,
        0.562108137758111, -0.785296641855087, -5.318335345720314]).reshape(D,ng)

    # Runs the PLDA trainer EM-steps (2 steps)
    # Defines base trainer and machine
    t = bob.trainer.PLDABaseTrainer(nf,ng)
    # TODO: constructor without argument
    m = bob.machine.PLDABaseMachine(D,nf,ng)

    # Calls the initialization methods and resets randomly initialized values
    # to new reference ones (to make the tests deterministic)
    t.initialization(m,l)
    m.sigma = sigma_init
    m.g = G_init
    m.f = F_init

    # E-step 1
    t.e_step(m,l)
    # Compares statistics to Prince matlab reference
    self.assertTrue(equals(t.z_first_order[0], z_first_order_a_1, 1e-10))
    self.assertTrue(equals(t.z_first_order[1], z_first_order_b_1, 1e-10))
    self.assertTrue(equals(t.z_second_order_sum, z_second_order_sum_1, 1e-10))

    # M-step 1
    t.m_step(m,l)
    # Compares F, G and sigma to Prince matlab reference
    self.assertTrue(equals(m.f, F_1, 1e-10))
    self.assertTrue(equals(m.g, G_1, 1e-10))
    self.assertTrue(equals(m.sigma, sigma_1, 1e-10))

    # E-step 2
    t.e_step(m,l)
    # Compares statistics to Prince matlab reference
    self.assertTrue(equals(t.z_first_order[0], z_first_order_a_2, 1e-10))
    self.assertTrue(equals(t.z_first_order[1], z_first_order_b_2, 1e-10))
    self.assertTrue(equals(t.z_second_order_sum, z_second_order_sum_2, 1e-10))

    # M-step 2
    t.m_step(m,l)
    # Compares F, G and sigma to Prince matlab reference
    self.assertTrue(equals(m.f, F_2, 1e-10))
    self.assertTrue(equals(m.g, G_2, 1e-10))
    self.assertTrue(equals(m.sigma, sigma_2, 1e-10))


  def test02_plda_likelihood(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    D = 7
    nf = 2
    ng = 3

    # initial values for F, G and sigma
    G_init=numpy.array([-1.1424, -0.5044, -0.1917,
      -0.6249,  0.1021, -0.8658,
      -1.1687,  1.1963,  0.1807,
      0.3926,  0.1203,  1.2665,
      1.3018, -1.0368, -0.2512,
      -0.5936, -0.8571, -0.2046,
      0.4364, -0.1699, -2.2015]).reshape(D,ng)
    # F <-> PCA on G
    F_init=numpy.array([-0.054222647972093, -0.000000000783146, 
      0.596449127693018,  0.000000006265167, 
      0.298224563846509,  0.000000003132583, 
      0.447336845769764,  0.000000009397750, 
      -0.108445295944185, -0.000000001566292, 
      -0.501559493741856, -0.000000006265167, 
      -0.298224563846509, -0.000000003132583]).reshape(D,nf)
    sigma_init = 0.01 * numpy.ones((D,), 'float64')
    mean_zero = numpy.zeros((D,), 'float64')

    # base machine
    mb = bob.machine.PLDABaseMachine(D,nf,ng)
    mb.sigma = sigma_init
    mb.g = G_init
    mb.f = F_init
    mb.mu = mean_zero

    # Data for likelihood computation
    x1 = numpy.array([0.8032, 0.3503, 0.4587, 0.9511, 0.1330, 0.0703, 0.7061])
    x2 = numpy.array([0.9317, 0.1089, 0.6517, 0.1461, 0.6940, 0.6256, 0.0437])
    x3 = numpy.array([0.7979, 0.9862, 0.4367, 0.3447, 0.0488, 0.2252, 0.5810])
    X = numpy.ndarray((3,D), 'float64')
    X[0,:] = x1
    X[1,:] = x2
    X[2,:] = x3
    a = []
    a.append(x1)
    a.append(x2)
    a.append(x3)
    a = numpy.array(a)

    # reference likelihood from Prince implementation
    ll_ref = -182.8880743535197

    # machine
    m = bob.machine.PLDAMachine(mb)
    ll = m.compute_likelihood(X)
    self.assertTrue(abs(ll - ll_ref) < 1e-10)

    # log likelihood ratio
    Y = numpy.ndarray((2,D), 'float64')
    Y[0,:] = x1
    Y[1,:] = x2
    Z = numpy.ndarray((1,D), 'float64')
    Z[0,:] = x3
    llX = m.compute_likelihood(X)
    llY = m.compute_likelihood(Y)
    llZ = m.compute_likelihood(Z)
    # reference obtained by computing the likelihood of [x1,x2,x3], [x1,x2] 
    # and [x3] separately
    llr_ref = -4.43695386675
    self.assertTrue(abs((llX - (llY + llZ)) - llr_ref) < 1e-10)


  def test03_plda_enrollment(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    D = 7
    nf = 2
    ng = 3

    # initial values for F, G and sigma
    G_init=numpy.array([-1.1424, -0.5044, -0.1917,
      -0.6249,  0.1021, -0.8658,
      -1.1687,  1.1963,  0.1807,
      0.3926,  0.1203,  1.2665,
      1.3018, -1.0368, -0.2512,
      -0.5936, -0.8571, -0.2046,
      0.4364, -0.1699, -2.2015]).reshape(D,ng)
    # F <-> PCA on G
    F_init=numpy.array([-0.054222647972093, -0.000000000783146, 
      0.596449127693018,  0.000000006265167, 
      0.298224563846509,  0.000000003132583, 
      0.447336845769764,  0.000000009397750, 
      -0.108445295944185, -0.000000001566292, 
      -0.501559493741856, -0.000000006265167, 
      -0.298224563846509, -0.000000003132583]).reshape(D,nf)
    sigma_init = 0.01 * numpy.ones((D,), 'float64')
    mean_zero = numpy.zeros((D,), 'float64')

    # base machine
    mb = bob.machine.PLDABaseMachine(D,nf,ng)
    mb.sigma = sigma_init
    mb.g = G_init
    mb.f = F_init
    mb.mu = mean_zero

    # Data for likelihood computation
    x1 = numpy.array([0.8032, 0.3503, 0.4587, 0.9511, 0.1330, 0.0703, 0.7061])
    x2 = numpy.array([0.9317, 0.1089, 0.6517, 0.1461, 0.6940, 0.6256, 0.0437])
    x3 = numpy.array([0.7979, 0.9862, 0.4367, 0.3447, 0.0488, 0.2252, 0.5810])
    a_enrol = []
    a_enrol.append(x1)
    a_enrol.append(x2)
    a_enrol = numpy.array(a_enrol)

    # reference likelihood from Prince implementation
    ll_ref = -182.8880743535197

    # Computes the likelihood using x1 and x2 as enrollment samples
    # and x3 as a probe sample
    m = bob.machine.PLDAMachine(mb)
    t = bob.trainer.PLDATrainer(m)
    t.enrol(a_enrol)
    ll = m.compute_likelihood(x3)
    self.assertTrue(abs(ll - ll_ref) < 1e-10)

    # reference obtained by computing the likelihood of [x1,x2,x3], [x1,x2] 
    # and [x3] separately
    llr_ref = -4.43695386675
    llr = m.forward(x3)
    self.assertTrue(abs(llr - llr_ref) < 1e-10)
