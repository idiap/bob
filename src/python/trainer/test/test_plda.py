#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri 14 Oct 2011

"""Tests PLDA trainer
"""

import os, sys
import unittest
import torch
import random

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
    a = torch.io.Arrayset()
    a.append(torch.core.array.float64_1([1,2,3,4,5,6,7], (D,)))
    a.append(torch.core.array.float64_1([7,8,3,3,1,8,2], (D,)))
    a.append(torch.core.array.float64_1([3,2,1,4,5,1,7], (D,)))
    a.append(torch.core.array.float64_1([9,0,3,2,1,4,6], (D,)))
    # second identity (3 samples)
    b = torch.io.Arrayset()
    b.append(torch.core.array.float64_1([5,6,3,4,2,0,2], (D,)))
    b.append(torch.core.array.float64_1([1,7,8,9,4,4,8], (D,)))
    b.append(torch.core.array.float64_1([8,7,2,5,1,1,1], (D,)))
    # list of Arrayset (training data)
    l = [a,b]
    # initial values for F, G and sigma
    G_init=torch.core.array.float64_2([-1.1424, -0.5044, -0.1917,
                                       -0.6249,  0.1021, -0.8658,
                                       -1.1687,  1.1963,  0.1807,
                                        0.3926,  0.1203,  1.2665,
                                        1.3018, -1.0368, -0.2512,
                                       -0.5936, -0.8571, -0.2046,
                                        0.4364, -0.1699, -2.2015], (D,ng))
    # F <-> PCA on G
    F_init=torch.core.array.float64_2([-0.054222647972093, -0.000000000783146, 
                                        0.596449127693018,  0.000000006265167, 
                                        0.298224563846509,  0.000000003132583, 
                                        0.447336845769764,  0.000000009397750, 
                                       -0.108445295944185, -0.000000001566292, 
                                       -0.501559493741856, -0.000000006265167, 
                                       -0.298224563846509, -0.000000003132583], (D,nf))
    sigma_init = torch.core.array.float64_1((D,))
    sigma_init.fill(0.01)

    # Defines reference results based on Princes'matlab implementation
    # After 1 iteration
    z_first_order_a_1 = torch.core.array.float64_2(
      [-2.624115900658397, -0.000000034277848,  1.554823055585319,  0.627476234024656, -0.264705934182394,
       -2.624115900658397, -0.000000034277848, -2.703482671599357, -1.533283607433197,  0.553725774828231,
       -2.624115900658397, -0.000000034277848,  2.311647528461115,  1.266362142140170, -0.317378177105131,
       -2.624115900658397, -0.000000034277848, -1.163402640008200, -0.372604542926019,  0.025152800097991
      ], (4, nf+ng))
    z_first_order_b_1 = torch.core.array.float64_2(
      [ 3.494168818797438,  0.000000045643026,  0.111295550530958, -0.029241422535725,  0.257045446451067,
        3.494168818797438,  0.000000045643026,  1.102110715965762,  1.481232954001794, -0.970661225144399,
        3.494168818797438,  0.000000045643026, -1.212854031699468, -1.435946529317718,  0.717884143973377
      ], (3, nf+ng))
  
    z_second_order_sum_1 = torch.core.array.float64_2(
      [64.203518285366087,  0.000000747228248,  0.002703277337642,  0.078542842475345,  0.020894328259862,
        0.000000747228248,  6.999999999999980, -0.000000003955962,  0.000000002017232, -0.000000003741593,
        0.002703277337642, -0.000000003955962, 19.136889380923918, 11.860493771107487, -4.584339465366988,
        0.078542842475345,  0.000000002017232, 11.860493771107487,  8.771502339750128, -3.905706024997424,
        0.020894328259862, -0.000000003741593, -4.584339465366988, -3.905706024997424,  2.011924970338584
      ], (nf+ng, nf+ng))

    sigma_1 = torch.core.array.float64_1(
      [2.193659969999207, 3.748361365521041, 0.237835235737085, 0.558546035892629,
       0.209272700958400, 1.717782807724451, 0.248414618308223], (D,))

    F_1 = torch.core.array.float64_2(
      [-0.059083416465692,  0.000000000751007,
        0.600133217253169,  0.000000006957266,
        0.302789123922871,  0.000000000218947,
        0.454540641429714,  0.000000003342540,
       -0.106608957780613, -0.000000001641389,
       -0.494267694269430, -0.000000011059552,
       -0.295956102084270, -0.000000006718366], (D,nf))

    G_1 = torch.core.array.float64_2(
      [-1.836166150865047,  2.491475145758734,  5.095958946372235,
       -0.608732205531767, -0.618128420353493, -1.085423135463635,
       -0.697390472635929, -1.047900122276840, -6.080211153116984,
        0.769509301515319, -2.763610156675313, -5.972172587527176,
        1.332474692714491, -1.368103875407414, -2.096382536513033,
        0.304135903830416, -5.168096082564016, -9.604769461465978,
        0.597445549865284, -1.347101803379971, -5.900246013340080], (D,ng))

    # After 2 iterations
    z_first_order_a_2 = torch.core.array.float64_2(
      [-2.144344161196005, -0.000000027851878,  1.217776189037369,  0.232492571855061, -0.212892893868819,
       -2.144344161196005, -0.000000027851878, -2.382647766948079, -1.759951013670071,  0.587213207926731,
       -2.144344161196005, -0.000000027851878,  2.143294830538722,  0.909307594408923, -0.183752098508072,
       -2.144344161196005, -0.000000027851878, -0.662558006326892,  0.717992497547010, -0.202897892977004
      ], (4, nf+ng))
    z_first_order_b_2 = torch.core.array.float64_2(
      [ 2.695117129662246,  0.000000035005543, -0.156173294945791, -0.123083763746364,  0.271123341933619,
        2.695117129662246,  0.000000035005543,  0.690321563509753,  0.944473716646212, -0.850835940962492,
        2.695117129662246,  0.000000035005543, -0.930970138998433, -0.949736472690315,  0.594216348861889
      ], (3, nf+ng))
 
    z_second_order_sum_2 = torch.core.array.float64_2(
      [41.602421167226410,  0.000000449434708, -1.513391506933811, -0.477818674270533,  0.059260102368316,
        0.000000449434708,  7.000000000000005, -0.000000023255959, -0.000000005157439, -0.000000003230262,
       -1.513391506933810, -0.000000023255959, 14.399631061987494,  8.068678077509025, -3.227586434905497,
       -0.477818674270533, -0.000000005157439,  8.068678077509025,  7.263248678863863, -3.060665688064639,
        0.059260102368316, -0.000000003230262, -3.227586434905497, -3.060665688064639,  1.705174220723198
      ], (nf+ng, nf+ng))

    sigma_2 = torch.core.array.float64_1(
      [1.120493935052524, 1.777598857891599, 0.197579528599150, 0.407657093211478,
       0.166216300651473, 1.044336960403809, 0.287856936559308], (D,))
 
    F_2 = torch.core.array.float64_2(
      [-0.111956311978966,  0.000000000781025,
        0.702502767389263,  0.000000007683917,
        0.337823622542517,  0.000000000637302,
        0.551363737526339,  0.000000004854293,
       -0.096561040511417, -0.000000001716011,
       -0.661587484803602, -0.000000012394362,
       -0.346593051621620, -0.000000007134046], (D,nf))

    G_2 = torch.core.array.float64_2(
      [-2.266404374274820,  4.089199685832099,  7.023039382876370,
        0.094887459097613, -3.226829318470136, -3.452279917194724,
       -0.498398131733141, -1.651712333649899, -6.548008210704172,
        0.574932298590327, -2.198978667003715, -5.131253543126156,
        1.415857426810629, -1.627795701160212, -2.509013676007012,
       -0.543552834305580, -3.215063993186718, -7.006305082499653,
        0.562108137758111, -0.785296641855087, -5.318335345720314], (D,ng))

    # Runs the PLDA trainer EM-steps (2 steps)
    # Defines base trainer and machine
    t = torch.trainer.PLDABaseTrainer(nf,ng)
    # TODO: constructor without argument
    m = torch.machine.PLDABaseMachine(D,nf,ng)

    # Calls the initialization methods and resets randomly initialized values
    # to new reference ones (to make the tests deterministic)
    t.initialization(m,l)
    m.sigma = sigma_init
    m.G = G_init
    m.F = F_init

    # E-step 1
    t.eStep(m,l)
    # Compares statistics to Prince matlab reference
    self.assertTrue(equals(t.z_first_order[0], z_first_order_a_1, 1e-10))
    self.assertTrue(equals(t.z_first_order[1], z_first_order_b_1, 1e-10))
    self.assertTrue(equals(t.z_second_order_sum, z_second_order_sum_1, 1e-10))

    # M-step 1
    t.mStep(m,l)
    # Compares F, G and sigma to Prince matlab reference
    self.assertTrue(equals(m.F, F_1, 1e-10))
    self.assertTrue(equals(m.G, G_1, 1e-10))
    self.assertTrue(equals(m.sigma, sigma_1, 1e-10))

    # E-step 2
    t.eStep(m,l)
    # Compares statistics to Prince matlab reference
    self.assertTrue(equals(t.z_first_order[0], z_first_order_a_2, 1e-10))
    self.assertTrue(equals(t.z_first_order[1], z_first_order_b_2, 1e-10))
    self.assertTrue(equals(t.z_second_order_sum, z_second_order_sum_2, 1e-10))

    # M-step 2
    t.mStep(m,l)
    # Compares F, G and sigma to Prince matlab reference
    self.assertTrue(equals(m.F, F_2, 1e-10))
    self.assertTrue(equals(m.G, G_2, 1e-10))
    self.assertTrue(equals(m.sigma, sigma_2, 1e-10))


  def test02_plda_likelihood(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    D = 7
    nf = 2
    ng = 3

    # initial values for F, G and sigma
    G_init=torch.core.array.float64_2([-1.1424, -0.5044, -0.1917,
                                       -0.6249,  0.1021, -0.8658,
                                       -1.1687,  1.1963,  0.1807,
                                        0.3926,  0.1203,  1.2665,
                                        1.3018, -1.0368, -0.2512,
                                       -0.5936, -0.8571, -0.2046,
                                        0.4364, -0.1699, -2.2015], (D,ng))
    # F <-> PCA on G
    F_init=torch.core.array.float64_2([-0.054222647972093, -0.000000000783146, 
                                        0.596449127693018,  0.000000006265167, 
                                        0.298224563846509,  0.000000003132583, 
                                        0.447336845769764,  0.000000009397750, 
                                       -0.108445295944185, -0.000000001566292, 
                                       -0.501559493741856, -0.000000006265167, 
                                       -0.298224563846509, -0.000000003132583], (D,nf))
    sigma_init = torch.core.array.float64_1((D,))
    sigma_init.fill(0.01)
    mean_zero = torch.core.array.float64_1((D,))
    mean_zero.fill(0)

    # base machine
    mb = torch.machine.PLDABaseMachine(D,nf,ng)
    mb.sigma = sigma_init
    mb.G = G_init
    mb.F = F_init
    mb.mu = mean_zero

    # Data for likelihood computation
    x1 = torch.core.array.float64_1([0.8032, 0.3503, 0.4587, 0.9511, 0.1330, 0.0703, 0.7061], (D,))
    x2 = torch.core.array.float64_1([0.9317, 0.1089, 0.6517, 0.1461, 0.6940, 0.6256, 0.0437], (D,))
    x3 = torch.core.array.float64_1([0.7979, 0.9862, 0.4367, 0.3447, 0.0488, 0.2252, 0.5810], (D,))
    X = torch.core.array.float64_2((3,D))
    X[0,:] = x1
    X[1,:] = x2
    X[2,:] = x3
    a = torch.io.Arrayset()
    a.append(x1)
    a.append(x2)
    a.append(x3)

    # reference likelihood from Prince implementation
    ll_ref = -182.8880743535197

    # machine
    m = torch.machine.PLDAMachine(mb)
    ll = m.computeLikelihood(X)
    self.assertTrue(abs(ll - ll_ref) < 1e-10)

    # log likelihood ratio
    Y = torch.core.array.float64_2((2,D))
    Y[0,:] = x1
    Y[1,:] = x2
    Z = torch.core.array.float64_2((1,D))
    Z[0,:] = x3
    llX = m.computeLikelihood(X)
    llY = m.computeLikelihood(Y)
    llZ = m.computeLikelihood(Z)
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
    G_init=torch.core.array.float64_2([-1.1424, -0.5044, -0.1917,
                                       -0.6249,  0.1021, -0.8658,
                                       -1.1687,  1.1963,  0.1807,
                                        0.3926,  0.1203,  1.2665,
                                        1.3018, -1.0368, -0.2512,
                                       -0.5936, -0.8571, -0.2046,
                                        0.4364, -0.1699, -2.2015], (D,ng))
    # F <-> PCA on G
    F_init=torch.core.array.float64_2([-0.054222647972093, -0.000000000783146, 
                                        0.596449127693018,  0.000000006265167, 
                                        0.298224563846509,  0.000000003132583, 
                                        0.447336845769764,  0.000000009397750, 
                                       -0.108445295944185, -0.000000001566292, 
                                       -0.501559493741856, -0.000000006265167, 
                                       -0.298224563846509, -0.000000003132583], (D,nf))
    sigma_init = torch.core.array.float64_1((D,))
    sigma_init.fill(0.01)
    mean_zero = torch.core.array.float64_1((D,))
    mean_zero.fill(0)

    # base machine
    mb = torch.machine.PLDABaseMachine(D,nf,ng)
    mb.sigma = sigma_init
    mb.G = G_init
    mb.F = F_init
    mb.mu = mean_zero

    # Data for likelihood computation
    x1 = torch.core.array.float64_1([0.8032, 0.3503, 0.4587, 0.9511, 0.1330, 0.0703, 0.7061], (D,))
    x2 = torch.core.array.float64_1([0.9317, 0.1089, 0.6517, 0.1461, 0.6940, 0.6256, 0.0437], (D,))
    x3 = torch.core.array.float64_1([0.7979, 0.9862, 0.4367, 0.3447, 0.0488, 0.2252, 0.5810], (D,))
    a_enrol = torch.io.Arrayset()
    a_enrol.append(x1)
    a_enrol.append(x2)

    # reference likelihood from Prince implementation
    ll_ref = -182.8880743535197

    # Computes the likelihood using x1 and x2 as enrollment samples
    # and x3 as a probe sample
    m = torch.machine.PLDAMachine(mb)
    t = torch.trainer.PLDATrainer(m)
    t.enrol(a_enrol)
    ll = m.computeLikelihood(x3)
    self.assertTrue(abs(ll - ll_ref) < 1e-10)

    # reference obtained by computing the likelihood of [x1,x2,x3], [x1,x2] 
    # and [x3] separately
    llr_ref = -4.43695386675
    llr = m.forward(x3)
    self.assertTrue(abs(llr - llr_ref) < 1e-10)


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
