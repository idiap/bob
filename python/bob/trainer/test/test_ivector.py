#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Tests the I-Vector trainer
"""

import unittest
import bob, numpy, numpy.linalg, numpy.random

### Test class inspired by an implementation of Chris McCool
### Chris McCool (chris.mccool@nicta.com.au)
class IVectorTrainerPy():
  """An IVector extractor"""

  def __init__(self, convergence_threshold=0.001, max_iterations=10,
      compute_likelihood=False, sigma_update=False, variance_floor=1e-5):
    self.m_convergence_threshold = convergence_threshold
    self.m_max_iterations = max_iterations
    self.m_compute_likelihood = compute_likelihood
    self.m_sigma_update = sigma_update
    self.m_variance_floor = variance_floor

  def initialize(self, machine, data):
    ubm = machine.ubm
    self.m_dim_c = ubm.dim_c
    self.m_dim_d = ubm.dim_d
    self.m_dim_t = machine.t.shape[1]
    self.m_meansupervector = ubm.mean_supervector
    t = numpy.random.randn(self.m_dim_c*self.m_dim_d, self.m_dim_t)
    machine.t = t
    machine.sigma = machine.ubm.variance_supervector

  def e_step(self, machine, data):
    n_samples = len(data)
    self.m_acc_Nij_Sigma_wij2  = {}
    self.m_acc_Fnorm_Sigma_wij = {}
    self.m_acc_Snorm = numpy.zeros(shape=(self.m_dim_c*self.m_dim_d,), dtype=numpy.float64)
    self.m_N = numpy.zeros(shape=(self.m_dim_c,), dtype=numpy.float64)

    for c in range(self.m_dim_c):
      self.m_acc_Nij_Sigma_wij2[c]  = numpy.zeros(shape=(self.m_dim_t,self.m_dim_t), dtype=numpy.float64)
      self.m_acc_Fnorm_Sigma_wij[c] = numpy.zeros(shape=(self.m_dim_d,self.m_dim_t), dtype=numpy.float64)

    for n in range(n_samples):
      Nij = data[n].n
      Fij = data[n].sum_px
      Sij = data[n].sum_pxx

      # Estimate latent variables
      TtSigmaInv_Fnorm = machine.__compute_TtSigmaInvFnorm__(data[n])
      I_TtSigmaInvNT = machine.__compute_Id_TtSigmaInvT__(data[n])

      Fnorm = numpy.zeros(shape=(self.m_dim_c*self.m_dim_d,), dtype=numpy.float64)
      Snorm = numpy.zeros(shape=(self.m_dim_c*self.m_dim_d,), dtype=numpy.float64)

      # Compute normalized statistics
      for c in range(self.m_dim_c):
        start            = c*self.m_dim_d
        end              = (c+1)*self.m_dim_d

        Fc               = Fij[c,:]
        Sc               = Sij[c,:]
        mc               = self.m_meansupervector[start:end]

        Fc_mc            = Fc * mc
        Nc_mc_mcT        = Nij[c] * mc * mc

        Fnorm[start:end] = Fc - Nij[c] * mc
        Snorm[start:end] = Sc - (2 * Fc_mc) + Nc_mc_mcT

      # Latent variables
      I_TtSigmaInvNT_inv = numpy.linalg.inv(I_TtSigmaInvNT)
      E_w_ij             = numpy.dot(I_TtSigmaInvNT_inv, TtSigmaInv_Fnorm)
      E_w_ij2            = I_TtSigmaInvNT_inv + numpy.outer(E_w_ij, E_w_ij)

      # Do the accumulation for each component
      self.m_acc_Snorm   = self.m_acc_Snorm + Snorm    # (dim_c*dim_d)
      for c in range(self.m_dim_c):
        start            = c*self.m_dim_d
        end              = (c+1)*self.m_dim_d
        current_Fnorm    = Fnorm[start:end]            # (dim_d)
        self.m_acc_Nij_Sigma_wij2[c]  = self.m_acc_Nij_Sigma_wij2[c] + Nij[c] * E_w_ij2                    # (dim_t, dim_t)
        self.m_acc_Fnorm_Sigma_wij[c] = self.m_acc_Fnorm_Sigma_wij[c] + numpy.outer(current_Fnorm, E_w_ij) # (dim_d, dim_t)
        self.m_N[c]                   = self.m_N[c] + Nij[c]


  def m_step(self, machine, data):
    A = self.m_acc_Nij_Sigma_wij2

    T = numpy.zeros(shape=(self.m_dim_c*self.m_dim_d,self.m_dim_t), dtype=numpy.float64)
    Told = machine.t
    if self.m_sigma_update:
      sigma = numpy.zeros(shape=self.m_acc_Snorm.shape, dtype=numpy.float64)
    for c in range(self.m_dim_c):
      start = c*self.m_dim_d;
      end   = (c+1)*self.m_dim_d;
      # T update
      A     = self.m_acc_Nij_Sigma_wij2[c].transpose()
      B     = self.m_acc_Fnorm_Sigma_wij[c].transpose()
      if numpy.array_equal(A, numpy.zeros(A.shape)):
        X = numpy.zeros(shape=(self.m_dim_t,self.m_dim_d), dtype=numpy.float64)
      else:
        X = numpy.linalg.solve(A, B)
      T[start:end,:] = X.transpose()
      # Sigma update
      if self.m_sigma_update:
        Told_c           = Told[start:end,:].transpose()
        # warning: Use of the new T estimate! (revert second next line if you don't want that)
        Fnorm_Ewij_Tt    = numpy.diag(numpy.dot(self.m_acc_Fnorm_Sigma_wij[c], X))
        #Fnorm_Ewij_Tt = numpy.diag(numpy.dot(self.m_acc_Fnorm_Sigma_wij[c], Told_c))
        sigma[start:end] = (self.m_acc_Snorm[start:end] - Fnorm_Ewij_Tt) / self.m_N[c]
      
    machine.t = T
    if self.m_sigma_update:
      sigma[sigma < self.m_variance_floor] = self.m_variance_floor
      machine.sigma = sigma

  def finalize(self, machine, data):
    pass

  def train(self, machine, data):
    self.initialize(machine, data)
    average_output_previous   = -sys.maxint
    average_output            = -sys.maxint
    self.e_step(machine, data)
    
    i = 0
    while True:
      average_output_previous = average_output
      self.m_step(machine, data)
      self.e_step(machine, data)
      if(self.m_max_iterations > 0 and i+1 >= self.m_max_iterations):
        break
      i += 1


class IVectorTests(unittest.TestCase):

  def test01_trainer_nosigma(self):
    # Ubm
    ubm = bob.machine.GMMMachine(2,3)
    ubm.weights = numpy.array([0.4,0.6])
    ubm.means = numpy.array([[1.,7,4],[4,5,3]])
    ubm.variances = numpy.array([[0.5,1.,1.5],[1.,1.5,2.]])

    # Defines GMMStats
    gs1 = bob.machine.GMMStats(2,3)
    log_likelihood1 = -3. 
    T1 = 1 
    n1 = numpy.array([0.4, 0.6], numpy.float64)
    sumpx1 = numpy.array([[1., 2., 3.], [2., 4., 3.]], numpy.float64)
    sumpxx1 = numpy.array([[10., 20., 30.], [40., 50., 60.]], numpy.float64)
    gs1.log_likelihood = log_likelihood1
    gs1.t = T1
    gs1.n = n1
    gs1.sum_px = sumpx1
    gs1.sum_pxx = sumpxx1

    gs2 = bob.machine.GMMStats(2,3)
    log_likelihood2 = -4. 
    T2 = 1 
    n2 = numpy.array([0.2, 0.8], numpy.float64)
    sumpx2 = numpy.array([[2., 1., 3.], [3., 4.1, 3.2]], numpy.float64)
    sumpxx2 = numpy.array([[12., 15., 25.], [39., 51., 62.]], numpy.float64)
    gs2.log_likelihood = log_likelihood2
    gs2.t = T2
    gs2.n = n2
    gs2.sum_px = sumpx2
    gs2.sum_pxx = sumpxx2

    data = [gs1, gs2]


    acc_Nij_Sigma_wij2_ref1  = {0: numpy.array([[ 0.03202305, -0.02947769], [-0.02947769,  0.0561132 ]]),
                               1: numpy.array([[ 0.07953279, -0.07829414], [-0.07829414,  0.13814242]])}
    acc_Fnorm_Sigma_wij_ref1 = {0: numpy.array([[-0.29622691,  0.61411796], [ 0.09391764, -0.27955961], [-0.39014455,  0.89367757]]), 
                               1: numpy.array([[ 0.04695882, -0.13977981], [-0.05718673,  0.24159665], [-0.17098161,  0.47326585]])}
    acc_Snorm_ref1           = numpy.array([16.6, 22.4, 16.6, 61.4, 55., 97.4])
    N_ref1                   = numpy.array([0.6, 1.4]) 
    t_ref1                   = numpy.array([[  1.59543739, 11.78239235], [ -3.20130371, -6.66379081], [  4.79674111, 18.44618316],
                                            [ -0.91765407, -1.5319461 ], [  2.26805901,  3.03434944], [  2.76600031,  4.9935962 ]])

    acc_Nij_Sigma_wij2_ref2  = {0: numpy.array([[ 0.37558389, -0.15405228], [-0.15405228,  0.1421269 ]]),
                               1: numpy.array([[ 1.02076081, -0.57683953], [-0.57683953,  0.53912239]])}
    acc_Fnorm_Sigma_wij_ref2 = {0: numpy.array([[-1.1261668 ,  1.46496753], [-0.03579289, -0.37875811], [-1.09037391,  1.84372565]]),
                               1: numpy.array([[-0.01789645, -0.18937906], [ 0.35221084,  0.15854126], [-0.10004552,  0.72559036]])}
    acc_Snorm_ref2           = numpy.array([16.6, 22.4, 16.6, 61.4, 55., 97.4])
    N_ref2                   = numpy.array([0.6, 1.4])
    t_ref2                   = numpy.array([[  2.2133685,  12.70654597], [ -2.13959381, -4.98404887], [  4.35296231, 17.69059484],
                                            [ -0.54644055, -0.93594252], [  1.29308324,  1.67762053], [  1.67583072,  3.13894546]])
    acc_Nij_Sigma_wij2_ref = [acc_Nij_Sigma_wij2_ref1, acc_Nij_Sigma_wij2_ref2]
    acc_Fnorm_Sigma_wij_ref = [acc_Fnorm_Sigma_wij_ref1, acc_Fnorm_Sigma_wij_ref2]
    acc_Snorm_ref = [acc_Snorm_ref1, acc_Snorm_ref2] 
    N_ref = [N_ref1, N_ref2]
    t_ref = [t_ref1, t_ref2]

    # Python implementation
    # Machine
    m = bob.machine.IVectorMachine(ubm, 2)
    t = numpy.array([[1.,2],[4,1],[0,3],[5,8],[7,10],[11,1]])
    sigma = numpy.array([1.,2.,1.,3.,2.,4.])

    # Initialization
    trainer = IVectorTrainerPy()
    trainer.initialize(m, data)
    m.t = t
    m.sigma = sigma
    for it in range(2):
      # E-Step
      trainer.e_step(m, data)
      for k in acc_Nij_Sigma_wij2_ref[it]:
        self.assertTrue(numpy.allclose(acc_Nij_Sigma_wij2_ref[it][k], trainer.m_acc_Nij_Sigma_wij2[k], 1e-5))
      for k in acc_Fnorm_Sigma_wij_ref[it]:
        self.assertTrue(numpy.allclose(acc_Fnorm_Sigma_wij_ref[it][k], trainer.m_acc_Fnorm_Sigma_wij[k], 1e-5))
      self.assertTrue(numpy.allclose(acc_Snorm_ref[it], trainer.m_acc_Snorm, 1e-5))
      self.assertTrue(numpy.allclose(N_ref[it], trainer.m_N, 1e-5))

      # M-Step
      trainer.m_step(m, data)
      self.assertTrue(numpy.allclose(t_ref[it], m.t, 1e-5))

    # C++ implementation
    # Machine
    m = bob.machine.IVectorMachine(ubm, 2)

    # Initialization
    trainer = bob.trainer.IVectorTrainer()
    trainer.initialize(m, data)
    m.t = t 
    m.sigma = sigma
    for it in range(2):
      # E-Step
      trainer.e_step(m, data)
      for k in acc_Nij_Sigma_wij2_ref[it]:
        self.assertTrue(numpy.allclose(acc_Nij_Sigma_wij2_ref[it][k], trainer.acc_nij_wij2[k], 1e-5))
      for k in acc_Fnorm_Sigma_wij_ref[it]:
        self.assertTrue(numpy.allclose(acc_Fnorm_Sigma_wij_ref[it][k], trainer.acc_fnormij_wij[k], 1e-5))

      # M-Step
      trainer.m_step(m, data)
      self.assertTrue(numpy.allclose(t_ref[it], m.t, 1e-5))


  def test02_trainer_update_sigma(self):
    # Ubm
    dim_c = 2
    dim_d = 3
    ubm = bob.machine.GMMMachine(dim_c,dim_d)
    ubm.weights = numpy.array([0.4,0.6])
    ubm.means = numpy.array([[1.,7,4],[4,5,3]])
    ubm.variances = numpy.array([[0.5,1.,1.5],[1.,1.5,2.]])

    # Defines GMMStats
    gs1 = bob.machine.GMMStats(dim_c,dim_d)
    log_likelihood1 = -3. 
    T1 = 1 
    n1 = numpy.array([0.4, 0.6], numpy.float64)
    sumpx1 = numpy.array([[1., 2., 3.], [2., 4., 3.]], numpy.float64)
    sumpxx1 = numpy.array([[10., 20., 30.], [40., 50., 60.]], numpy.float64)
    gs1.log_likelihood = log_likelihood1
    gs1.t = T1
    gs1.n = n1
    gs1.sum_px = sumpx1
    gs1.sum_pxx = sumpxx1

    gs2 = bob.machine.GMMStats(dim_c,dim_d)
    log_likelihood2 = -4. 
    T2 = 1 
    n2 = numpy.array([0.2, 0.8], numpy.float64)
    sumpx2 = numpy.array([[2., 1., 3.], [3., 4.1, 3.2]], numpy.float64)
    sumpxx2 = numpy.array([[12., 15., 25.], [39., 51., 62.]], numpy.float64)
    gs2.log_likelihood = log_likelihood2
    gs2.t = T2
    gs2.n = n2
    gs2.sum_px = sumpx2
    gs2.sum_pxx = sumpxx2

    data = [gs1, gs2]

    # Reference values
    acc_Nij_Sigma_wij2_ref1  = {0: numpy.array([[ 0.03202305, -0.02947769], [-0.02947769,  0.0561132 ]]),
                                1: numpy.array([[ 0.07953279, -0.07829414], [-0.07829414,  0.13814242]])}
    acc_Fnorm_Sigma_wij_ref1 = {0: numpy.array([[-0.29622691,  0.61411796], [ 0.09391764, -0.27955961], [-0.39014455,  0.89367757]]), 
                                1: numpy.array([[ 0.04695882, -0.13977981], [-0.05718673,  0.24159665], [-0.17098161,  0.47326585]])}
    acc_Snorm_ref1           = numpy.array([16.6, 22.4, 16.6, 61.4, 55., 97.4])
    N_ref1                   = numpy.array([0.6, 1.4]) 
    t_ref1                   = numpy.array([[  1.59543739, 11.78239235], [ -3.20130371, -6.66379081], [  4.79674111, 18.44618316],
                                            [ -0.91765407, -1.5319461 ], [  2.26805901,  3.03434944], [  2.76600031,  4.9935962 ]])
    sigma_ref1               = numpy.array([ 16.39472121, 34.72955353,  3.3108037, 43.73496916, 38.85472445, 68.22116903])

    acc_Nij_Sigma_wij2_ref2  = {0: numpy.array([[ 0.50807426, -0.11907756], [-0.11907756,  0.12336544]]), 
                                1: numpy.array([[ 1.18602399, -0.2835859 ], [-0.2835859 ,  0.39440498]])}
    acc_Fnorm_Sigma_wij_ref2 = {0: numpy.array([[ 0.07221453,  1.1189786 ], [-0.08681275, -0.35396112], [ 0.15902728,  1.47293972]]), 
                                1: numpy.array([[-0.04340637, -0.17698056], [ 0.10662127,  0.21484933],[ 0.13116645,  0.64474271]])}
    acc_Snorm_ref2           = numpy.array([16.6, 22.4, 16.6, 61.4, 55., 97.4])
    N_ref2                   = numpy.array([0.6, 1.4])
    t_ref2                   = numpy.array([[  2.93105054, 11.89961223], [ -1.08988119, -3.92120757], [  4.02093173, 15.82081981],
                                            [ -0.17376634, -0.57366984], [  0.26585634,  0.73589952], [  0.60557877,   2.07014704]])
    sigma_ref2               = numpy.array([5.12154025e+00, 3.48623823e+01, 1.00000000e-05, 4.37792350e+01, 3.91525332e+01, 6.85613258e+01])

    acc_Nij_Sigma_wij2_ref = [acc_Nij_Sigma_wij2_ref1, acc_Nij_Sigma_wij2_ref2]
    acc_Fnorm_Sigma_wij_ref = [acc_Fnorm_Sigma_wij_ref1, acc_Fnorm_Sigma_wij_ref2]
    acc_Snorm_ref = [acc_Snorm_ref1, acc_Snorm_ref2] 
    N_ref = [N_ref1, N_ref2]
    t_ref = [t_ref1, t_ref2]
    sigma_ref = [sigma_ref1, sigma_ref2]


    # Python implementation
    # Machine
    m = bob.machine.IVectorMachine(ubm, 2)
    t = numpy.array([[1.,2],[4,1],[0,3],[5,8],[7,10],[11,1]])
    sigma = numpy.array([1.,2.,1.,3.,2.,4.])

    # Initialization
    trainer = IVectorTrainerPy(sigma_update=True)
    trainer.initialize(m, data)
    m.t = t
    m.sigma = sigma
    for it in range(2):
      # E-Step
      trainer.e_step(m, data)
      for k in acc_Nij_Sigma_wij2_ref[it]:
        self.assertTrue(numpy.allclose(acc_Nij_Sigma_wij2_ref[it][k], trainer.m_acc_Nij_Sigma_wij2[k], 1e-5))
      for k in acc_Fnorm_Sigma_wij_ref[it]:
        self.assertTrue(numpy.allclose(acc_Fnorm_Sigma_wij_ref[it][k], trainer.m_acc_Fnorm_Sigma_wij[k], 1e-5))
      self.assertTrue(numpy.allclose(acc_Snorm_ref[it], trainer.m_acc_Snorm, 1e-5))
      self.assertTrue(numpy.allclose(N_ref[it], trainer.m_N, 1e-5))

      # M-Step
      trainer.m_step(m, data)
      self.assertTrue(numpy.allclose(t_ref[it], m.t, 1e-5))
      self.assertTrue(numpy.allclose(sigma_ref[it], m.sigma, 1e-5))


    # C++ implementation
    # Machine
    m = bob.machine.IVectorMachine(ubm, 2)
    m.variance_threshold = 1e-5 

    # Initialization
    trainer = bob.trainer.IVectorTrainer(update_sigma=True)
    trainer.initialize(m, data)
    m.t = t 
    m.sigma = sigma
    for it in range(2):
      # E-Step
      trainer.e_step(m, data)
      for k in acc_Nij_Sigma_wij2_ref[it]:
        self.assertTrue(numpy.allclose(acc_Nij_Sigma_wij2_ref[it][k], trainer.acc_nij_wij2[k], 1e-5))
      for k in acc_Fnorm_Sigma_wij_ref[it]:
        self.assertTrue(numpy.allclose(acc_Fnorm_Sigma_wij_ref[it][k], trainer.acc_fnormij_wij[k], 1e-5))
      self.assertTrue(numpy.allclose(acc_Snorm_ref[it].reshape(dim_c,dim_d), trainer.acc_snormij, 1e-5))
      self.assertTrue(numpy.allclose(N_ref[it], trainer.acc_nij, 1e-5))

      # M-Step
      trainer.m_step(m, data)
      self.assertTrue(numpy.allclose(t_ref[it], m.t, 1e-5))
      self.assertTrue(numpy.allclose(sigma_ref[it], m.sigma, 1e-5))

