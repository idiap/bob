#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Sat Oct 22 23:01:09 2011 +0200
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

"""Tests PLDA machine
"""

import os, sys
import unittest
import bob
import random, tempfile, math
import numpy
import numpy.linalg

# Defines common variables globally
# Dimensionalities
C_dim_d = 7
C_dim_f = 2
C_dim_g = 3
# Values for F and G
C_G=numpy.array([-1.1424, -0.5044, -0.1917,
      -0.6249,  0.1021, -0.8658,
      -1.1687,  1.1963,  0.1807,
      0.3926,  0.1203,  1.2665,
      1.3018, -1.0368, -0.2512,
      -0.5936, -0.8571, -0.2046,
      0.4364, -0.1699, -2.2015], 'float64').reshape(C_dim_d, C_dim_g)
# F <-> PCA on G
C_F=numpy.array([-0.054222647972093, -0.000000000783146, 
      0.596449127693018,  0.000000006265167, 
      0.298224563846509,  0.000000003132583, 
      0.447336845769764,  0.000000009397750, 
      -0.108445295944185, -0.000000001566292, 
      -0.501559493741856, -0.000000006265167, 
      -0.298224563846509, -0.000000003132583], 'float64').reshape(C_dim_d, C_dim_f)

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

def compute_i_sigma(sigma):
  # Inverse of a diagonal matrix (represented by a 1D numpy array)
  return (1. / sigma)

def compute_alpha(G, sigma):
  # alpha = (Id + G^T.sigma^-1.G)^-1 = \mathcal{G}
  dim_g = G.shape[1]
  isigma = numpy.diag(compute_i_sigma(sigma))
  return numpy.linalg.inv(numpy.eye(dim_g) + numpy.dot(numpy.dot(G.transpose(), isigma), G))

def compute_beta(G, sigma):
  # beta = (sigma + G.G^T)^-1 = sigma^-1 - sigma^-1.G.alpha.G^T.sigma^-1 = \mathcal{S}
  isigma = numpy.diag(compute_i_sigma(sigma))
  gt_isigma = numpy.dot(G.transpose(), isigma)
  alpha = compute_alpha(G, sigma)
  return (isigma - numpy.dot(numpy.dot(gt_isigma.transpose(), alpha), gt_isigma))

def compute_gamma(F, G, sigma, a):
  # gamma_a = (Id + a.F^T.beta.F)^-1 = \mathcal{F}_{a}
  dim_f = F.shape[1]
  beta = compute_beta(G, sigma)
  return numpy.linalg.inv(numpy.eye(dim_f) + a * numpy.dot(numpy.dot(F.transpose(), beta), F))
 
def compute_ft_beta(F, G, sigma):
  # F^T.beta = F^T.\mathcal{S}
  beta = compute_beta(G, sigma)
  return numpy.dot(numpy.transpose(F), beta)

def compute_gt_i_sigma(G, sigma):
  # G^T.sigma^-1
  isigma = compute_i_sigma(sigma)
  return numpy.transpose(G) * isigma

def compute_logdet_alpha(G, sigma):
  # \log(\det(\alpha)) = \log(\det(\mathcal{G}))
  alpha = compute_alpha(G, sigma)
  return math.log(numpy.linalg.det(alpha))
  
def compute_logdet_sigma(sigma):
  # \log(\det(\sigma)) = \log(\det(\sigma)) = \log(\prod(\sigma_{i}))
  return math.log(numpy.prod(sigma))

def compute_loglike_constterm(F, G, sigma, a):
  # loglike_constterm[a] = a/2 * ( -D*\log(2*pi) -\log|\sigma| +\log|\alpha| +\log|\gamma_a|)
  gamma_a = compute_gamma(F, G, sigma, a)
  logdet_gamma_a = math.log(abs(numpy.linalg.det(gamma_a)))
  ah = a/2.
  dim_d =  F.shape[0]
  logdet_sigma = compute_logdet_sigma(sigma)
  logdet_alpha = compute_logdet_alpha(G, sigma)
  res = -ah*dim_d*math.log(2*math.pi) - ah*logdet_sigma + ah*logdet_alpha + logdet_gamma_a/2.
  return res;

def compute_log_likelihood_point_estimate(observation, mu, F, G, sigma, hi, wij):
  """ 
  This function computes p(x_{ij} | h_{i}, w_{ij}, \Theta), which is given by
    N_{x}[\mu + Fh_{i} + Gw_{ij} + epsilon_{ij}, \Sigma], N_{x} being a 
    Gaussian distribution. As it returns the corresponding log likelihood, 
    this is given by the sum of the following three terms:
      C1 = -dim_d/2 log(2pi)               
      C2 = -1/2 log(det(\Sigma))
      C3 = -1/2 (x_{ij}-\mu-Fh_{i}-Gw_{ij})^{T}\Sigma^{-1}(x_{ij}-\mu-Fh_{i}-Gw_{ij})
  """
  
  ### Pre-computes some of the constants
  dim_d          = observation.shape[0]             # A scalar
  log_2pi        = numpy.log(2. * numpy.pi);        # A scalar
  C1             = -(dim_d / 2.) * log_2pi;         # A scalar
  C2             = -(1. / 2.) * numpy.sum( numpy.log(sigma) ); # (dim_d, 1)
  
  ### Subtract the identity and session components from the observed vector.    
  session_plus_identity  = numpy.dot(F, hi) + numpy.dot(G, wij);
  normalised_observation = numpy.reshape(observation - mu - session_plus_identity, (dim_d,1)); 
  ### Now calculate C3
  sigma_inverse  = numpy.reshape(1. / sigma, (dim_d,1));                      # (dim_d, 1)
  C3             = -(1. / 2.) * numpy.sum(normalised_observation * sigma_inverse * normalised_observation);
  
  ### Returns the log likelihood
  log_likelihood     = C1 + C2 + C3; 
  return (log_likelihood);


def compute_log_likelihood(observations, mu, F, G, sigma):
  """
  This function computes the log-likelihood of the observations given the parameters 
  of the PLDA model. This is done by fulling integrating out the latent variables.
  """
  # Work out the number of samples that we have and normalise the data.
  J_i                = observations.shape[0];                  # An integer > 0
  norm_observations  = observations - numpy.tile(mu, [J_i,1]);        # (J_i, D_x)

  # There are three terms that need to be computed: C1, C2 and C3

  # 1. Computes C1
  # C1 = - J_{i} * dim_d/2 log(2*pi)
  dim_d          = observations.shape[1]             # A scalar
  dim_f          = F.shape[1]
  log_2pi        = numpy.log(2. * numpy.pi);        # A scalar
  C1             = - J_i * (dim_d / 2.) * log_2pi;         # A scalar

  # 2. Computes C2
  # C2 = - J_i/2 * [log(det(sigma)) - log(det(alpha^-1))] + log(det(gamma_{J_i}))/2
  ld_sigma = compute_logdet_sigma(sigma)
  ld_alpha = compute_logdet_alpha(G, sigma)
  gamma = compute_gamma(F, G, sigma, J_i)
  ld_gamma = math.log(numpy.linalg.det(gamma))
  C2 = - J_i/2.*(ld_sigma - ld_alpha)  + ld_gamma/2.

  # 3. Computes C3
  # This is a quadratic part and consists of 
  # C3   = -0.5 * sum x^T beta x + 0.5 * Quadratic term in x
  # C3   = -0.5 * (C3a - C3b)
  C3a                  = 0.0;
  C3b_sum_part         = numpy.zeros((dim_f,1));
  isigma               = numpy.diag(compute_i_sigma(sigma))
  beta                 = compute_beta(G, sigma)
  ft_beta              = numpy.dot(numpy.transpose(F), beta)
  for j in range(0, J_i):
    ### Calculations for C3a
    current_vector           = numpy.reshape(norm_observations[j,:], (dim_d,1));  # (D_x, 1)
    vector_E                 = numpy.dot(beta, current_vector);                   # (D_x, 1)
    current_result           = numpy.dot(current_vector.transpose(), vector_E);   # A floating point value
    C3a                      = C3a + current_result[0][0];                        # A floating point value
    ### Calculations for C3b
    C3b_sum_part             = C3b_sum_part + numpy.dot(ft_beta, current_vector);  # (nf, 1)

  ### Final calculations for C3b, using the matrix gamma_{J_i}
  C3b                        = numpy.dot(numpy.dot(C3b_sum_part.transpose(), gamma), C3b_sum_part);
  C3                         = -0.5 * (C3a - C3b[0][0]);

  return C1 + C2 + C3


class PLDAMachineTest(unittest.TestCase):
  """Performs various PLDA machine tests."""

  def test01_plda_basemachine(self):
    # Data used for performing the tests
    sigma = numpy.ndarray(C_dim_d, 'float64')
    sigma.fill(0.01)
    mu = numpy.ndarray(C_dim_d, 'float64')
    mu.fill(0)

    # Defines reference results based on matlab
    alpha_ref = numpy.array([ 0.002189051545735,  0.001127099941432,
      -0.000145483208153, 0.001127099941432,  0.003549267943741,
      -0.000552001405453, -0.000145483208153, -0.000552001405453,
      0.001440505362615], 'float64').reshape(C_dim_g, C_dim_g)
    beta_ref  = numpy.array([ 50.587191765140361, -14.512478352504877,
      -0.294799164567830,  13.382002504394316,  9.202063877660278,
      -43.182264846086497,  11.932345916716455, -14.512478352504878,
      82.320149045633045, -12.605578822979698,  19.618675892079366,
      13.033691341150439,  -8.004874490989799, -21.547363307109187,
      -0.294799164567832, -12.605578822979696,  52.123885798398241,
      4.363739008635009, 44.847177605628545,  16.438137537463710,
      5.137421840557050, 13.382002504394316,  19.618675892079366,
      4.363739008635011,  75.070401560513488, -4.515472972526140,
      9.752862741017488,  34.196127678931106, 9.202063877660285,
      13.033691341150439,  44.847177605628552,  -4.515472972526142,
      56.189416227691098,  -7.536676357632515, -10.555735414707383,
      -43.182264846086497,  -8.004874490989799,  16.438137537463703,
      9.752862741017490, -7.536676357632518,  56.430571485722126,
      9.471758169835317, 11.932345916716461, -21.547363307109187,
      5.137421840557051,  34.196127678931099, -10.555735414707385,
      9.471758169835320,  27.996266602110637], 'float64').reshape(C_dim_d, C_dim_d)
    gamma3_ref = numpy.array([ 0.005318799462241, -0.000000012993151,
      -0.000000012993151,  0.999999999999996], 'float64').reshape(C_dim_f, C_dim_f)

    # Constructor tests
    m = bob.machine.PLDABaseMachine()
    self.assertTrue(m.dim_d == 0)
    self.assertTrue(m.dim_f == 0)
    self.assertTrue(m.dim_g == 0)
    del m
    m = bob.machine.PLDABaseMachine(C_dim_d, C_dim_f, C_dim_g)
    self.assertTrue(m.dim_d == C_dim_d)
    self.assertTrue(m.dim_f == C_dim_f)
    self.assertTrue(m.dim_g == C_dim_g)
    self.assertTrue(equals(m.variance_thresholds, numpy.zeros((C_dim_d,),dtype=numpy.float64), 1e-10))
    del m
    m = bob.machine.PLDABaseMachine(C_dim_d, C_dim_f, C_dim_g, 1e-2)
    self.assertTrue(m.dim_d == C_dim_d)
    self.assertTrue(m.dim_f == C_dim_f)
    self.assertTrue(m.dim_g == C_dim_g)
    self.assertTrue(equals(m.variance_thresholds, 1e-2*numpy.ones((C_dim_d,),dtype=numpy.float64), 1e-10))
    del m

    # Defines base machine
    m = bob.machine.PLDABaseMachine()
    m.resize(C_dim_d, C_dim_f, C_dim_g)
    # Sets the current mu, F, G and sigma 
    m.mu = mu
    m.f = C_F
    m.g = C_G
    m.sigma = sigma
    gamma3 = m.get_add_gamma(3).copy()
    constTerm3 = m.get_add_log_like_const_term(3)
    
    # Compares precomputed values to matlab reference
    self.assertTrue(equals(m.__alpha__, alpha_ref, 1e-10))
    self.assertTrue(equals(m.__beta__, beta_ref, 1e-10))
    self.assertTrue(equals(gamma3, gamma3_ref, 1e-10))

    # Compares precomputed values to the ones returned by python implementation
    self.assertTrue(equals(m.__isigma__, compute_i_sigma(sigma), 1e-10))
    self.assertTrue(equals(m.__alpha__, compute_alpha(C_G,sigma), 1e-10))
    self.assertTrue(equals(m.__beta__, compute_beta(C_G,sigma), 1e-10))
    self.assertTrue(equals(m.get_add_gamma(3), compute_gamma(C_F,C_G,sigma,3), 1e-10))
    self.assertTrue(m.has_gamma(3))
    self.assertTrue(equals(m.get_gamma(3), compute_gamma(C_F,C_G,sigma,3), 1e-10))
    self.assertTrue(equals(m.__ft_beta__, compute_ft_beta(C_F,C_G,sigma), 1e-10))
    self.assertTrue(equals(m.__gt_i_sigma__, compute_gt_i_sigma(C_G,sigma), 1e-10))
    self.assertTrue(math.fabs(m.__logdet_alpha__ - compute_logdet_alpha(C_G,sigma)) < 1e-10)
    self.assertTrue(math.fabs(m.__logdet_sigma__ - compute_logdet_sigma(sigma)) < 1e-10)
    self.assertTrue(abs(m.get_add_log_like_const_term(3) - compute_loglike_constterm(C_F,C_G,sigma,3)) < 1e-10)
    self.assertTrue(m.has_log_like_const_term(3))
    self.assertTrue(abs(m.get_log_like_const_term(3) - compute_loglike_constterm(C_F,C_G,sigma,3)) < 1e-10)

    # Defines base machine
    del m
    m = bob.machine.PLDABaseMachine(C_dim_d, C_dim_f, C_dim_g)
    # Sets the current mu, F, G and sigma 
    m.mu = mu
    m.f = C_F
    m.g = C_G
    m.sigma = sigma
    gamma3 = m.get_add_gamma(3).copy()
    constTerm3 = m.get_add_log_like_const_term(3)
    
    # Compares precomputed values to matlab reference
    self.assertTrue(equals(m.__alpha__, alpha_ref, 1e-10))
    self.assertTrue(equals(m.__beta__, beta_ref, 1e-10))
    self.assertTrue(equals(gamma3, gamma3_ref, 1e-10))

    # values before being saved
    isigma = m.__isigma__.copy()
    alpha = m.__alpha__.copy()
    beta = m.__beta__.copy()
    FtBeta = m.__ft_beta__.copy()
    GtISigma = m.__gt_i_sigma__.copy()
    logdetAlpha = m.__logdet_alpha__
    logdetSigma = m.__logdet_sigma__

    # Saves to file, loads and compares to original
    filename = str(tempfile.mkstemp(".hdf5")[1])
    m.save(bob.io.HDF5File(filename, 'w'))
    m_loaded = bob.machine.PLDABaseMachine(bob.io.HDF5File(filename))

    # Compares the values loaded with the former ones
    self.assertTrue(m_loaded == m)
    self.assertFalse(m_loaded != m)
    self.assertTrue(equals(m_loaded.mu, mu, 1e-10))
    self.assertTrue(equals(m_loaded.f, C_F, 1e-10))
    self.assertTrue(equals(m_loaded.g, C_G, 1e-10))
    self.assertTrue(equals(m_loaded.sigma, sigma, 1e-10))
    self.assertTrue(equals(m_loaded.__isigma__, isigma, 1e-10))
    self.assertTrue(equals(m_loaded.__alpha__, alpha, 1e-10))
    self.assertTrue(equals(m_loaded.__beta__, beta, 1e-10))
    self.assertTrue(equals(m_loaded.__ft_beta__, FtBeta, 1e-10))
    self.assertTrue(equals(m_loaded.__gt_i_sigma__, GtISigma, 1e-10))
    self.assertTrue(abs(m_loaded.__logdet_alpha__ - logdetAlpha) < 1e-10)
    self.assertTrue(abs(m_loaded.__logdet_sigma__ - logdetSigma) < 1e-10)
    self.assertTrue(m_loaded.has_gamma(3))
    self.assertTrue(equals(m_loaded.get_gamma(3), gamma3_ref, 1e-10))
    self.assertTrue(equals(m_loaded.get_add_gamma(3), gamma3_ref, 1e-10))
    self.assertTrue(m_loaded.has_log_like_const_term(3))
    self.assertTrue(abs(m_loaded.get_add_log_like_const_term(3) - constTerm3) < 1e-10)

    # Compares the values loaded with the former ones when copying
    m_copy = bob.machine.PLDABaseMachine(m_loaded)
    self.assertTrue(m_loaded == m_copy)
    self.assertFalse(m_loaded != m_copy)
    # Test clear_maps method
    self.assertTrue(m_copy.has_gamma(3))
    self.assertTrue(m_copy.has_log_like_const_term(3))
    m_copy.clear_maps()
    self.assertFalse(m_copy.has_gamma(3))
    self.assertFalse(m_copy.has_log_like_const_term(3))
    
    # Check variance flooring thresholds-related methods
    v_zo = numpy.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    v_zzo = numpy.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    m_copy.variance_thresholds = v_zo
    self.assertFalse(m_loaded == m_copy)
    self.assertTrue(m_loaded != m_copy)
    m_copy.variance_thresholds = v_zzo
    m_copy.sigma = v_zo
    self.assertTrue(equals(m_copy.sigma, v_zo, 1e-10))
    m_copy.variance_thresholds = v_zo
    m_copy.sigma = v_zzo
    self.assertTrue(equals(m_copy.sigma, v_zo, 1e-10))
    m_copy.variance_thresholds = v_zzo
    m_copy.sigma = v_zzo
    self.assertTrue(equals(m_copy.sigma, v_zzo, 1e-10))
    m_copy.variance_thresholds = v_zo
    self.assertTrue(equals(m_copy.sigma, v_zo, 1e-10)) 
   
    # Clean-up
    os.unlink(filename)

  def test02_plda_basemachine_loglikelihood_pointestimate(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    sigma = numpy.ndarray(C_dim_d, 'float64')
    sigma.fill(0.01)
    mu = numpy.ndarray(C_dim_d, 'float64')
    mu.fill(0)
    xij = numpy.array([0.7, 1.3, 2.5, 0.3, 1.3, 2.7, 0.9])
    hi = numpy.array([-0.5, 0.5])
    wij = numpy.array([-0.1, 0.2, 0.3])
    
    m = bob.machine.PLDABaseMachine(C_dim_d, C_dim_f, C_dim_g)
    # Sets the current mu, F, G and sigma 
    m.mu = mu
    m.f = C_F
    m.g = C_G
    m.sigma = sigma

    self.assertTrue(equals(m.compute_log_likelihood_point_estimate(xij, hi, wij), 
                           compute_log_likelihood_point_estimate(xij, mu, C_F, C_G, sigma, hi, wij), 1e-6))

  def test03_plda_machine(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    sigma = numpy.ndarray(C_dim_d, 'float64')
    sigma.fill(0.01)
    mu = numpy.ndarray(C_dim_d, 'float64')
    mu.fill(0)

    # Defines base machine
    mb = bob.machine.PLDABaseMachine(C_dim_d, C_dim_f, C_dim_g)
    # Sets the current mu, F, G and sigma 
    mb.mu = mu
    mb.f = C_F
    mb.g = C_G
    mb.sigma = sigma

    # Test constructors and dim getters
    m = bob.machine.PLDAMachine(mb)
    self.assertTrue(m.dim_d == C_dim_d)
    self.assertTrue(m.dim_f == C_dim_f)
    self.assertTrue(m.dim_g == C_dim_g)

    m0 = bob.machine.PLDAMachine()
    m0.plda_base = mb
    self.assertTrue(m0.dim_d == C_dim_d)
    self.assertTrue(m0.dim_f == C_dim_f)
    self.assertTrue(m0.dim_g == C_dim_g)
 
    # Defines machine
    n_samples = 2
    WSumXitBetaXi = 0.37
    weightedSum = numpy.array([1.39,0.54], 'float64')
    log_likelihood = -0.22

    m.n_samples = n_samples
    m.w_sum_xit_beta_xi = WSumXitBetaXi
    m.weighted_sum = weightedSum
    m.log_likelihood = log_likelihood

    gamma3 = m.get_add_gamma(3).copy()
    constTerm3 = m.get_add_log_like_const_term(3)

    # Saves to file, loads and compares to original
    filename = str(tempfile.mkstemp(".hdf5")[1])
    m.save(bob.io.HDF5File(filename, 'w'))
    m_loaded = bob.machine.PLDAMachine(bob.io.HDF5File(filename))
    m_loaded.plda_base = mb

    # Compares the values loaded with the former ones
    self.assertTrue(m_loaded == m)
    self.assertFalse(m_loaded != m)
    self.assertTrue(abs(m_loaded.n_samples - n_samples) < 1e-10)
    self.assertTrue(abs(m_loaded.w_sum_xit_beta_xi - WSumXitBetaXi) < 1e-10)
    self.assertTrue(equals(m_loaded.weighted_sum, weightedSum, 1e-10))
    self.assertTrue(abs(m_loaded.log_likelihood - log_likelihood) < 1e-10) 
    self.assertTrue(m_loaded.has_gamma(3))
    self.assertTrue(equals(m_loaded.get_add_gamma(3), gamma3, 1e-10))
    self.assertTrue(equals(m_loaded.get_gamma(3), gamma3, 1e-10))
    self.assertTrue(m_loaded.has_log_like_const_term(3))
    self.assertTrue(abs(m_loaded.get_add_log_like_const_term(3) - constTerm3) < 1e-10)
    self.assertTrue(abs(m_loaded.get_log_like_const_term(3) - constTerm3) < 1e-10)

    # Test clear_maps method
    self.assertTrue(m_loaded.has_gamma(3))
    self.assertTrue(m_loaded.has_log_like_const_term(3))
    m_loaded.clear_maps()
    self.assertFalse(m_loaded.has_gamma(3))
    self.assertFalse(m_loaded.has_log_like_const_term(3))
 
    # Clean-up
    os.unlink(filename)

  def test04_plda_machine_log_likelihood(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    sigma = numpy.ndarray(C_dim_d, 'float64')
    sigma.fill(0.01)
    mu = numpy.ndarray(C_dim_d, 'float64')
    mu.fill(0)

    # Defines base machine
    mb = bob.machine.PLDABaseMachine(C_dim_d, C_dim_f, C_dim_g)
    # Sets the current mu, F, G and sigma 
    mb.mu = mu
    mb.f = C_F
    mb.g = C_G
    mb.sigma = sigma

    # Defines machine
    m = bob.machine.PLDAMachine(mb)

    # Defines (random) samples and check compute_log_likelihood method
    ar_e = numpy.random.randn(2,C_dim_d)
    ar_p = numpy.random.randn(C_dim_d)
    ar_s = numpy.vstack([ar_e, ar_p])
    self.assertTrue(abs(m.compute_log_likelihood(ar_s, False) - compute_log_likelihood(ar_s, mu, C_F, C_G, sigma)) < 1e-10)
    ar_p2d = numpy.reshape(ar_p, (1,C_dim_d))
    self.assertTrue(abs(m.compute_log_likelihood(ar_p, False) - compute_log_likelihood(ar_p2d, mu, C_F, C_G, sigma)) < 1e-10)

    # Defines (random) samples and check forward method
    ar2_e = numpy.random.randn(4,C_dim_d)
    ar2_p = numpy.random.randn(C_dim_d)
    ar2_s = numpy.vstack([ar2_e, ar2_p])
    m.log_likelihood = m.compute_log_likelihood(ar2_e, False)
    llr = m.compute_log_likelihood(ar2_s, True) - (m.compute_log_likelihood(ar2_s, False) + m.log_likelihood)
    self.assertTrue(abs(m.forward(ar2_s) - llr) < 1e-10)
    ar2_p2d = numpy.random.randn(3,C_dim_d)
    ar2_s2d = numpy.vstack([ar2_e, ar2_p2d])
    llr2d = m.compute_log_likelihood(ar2_s2d, True) - (m.compute_log_likelihood(ar2_s2d, False) + m.log_likelihood)
    self.assertTrue(abs(m.forward(ar2_s2d) - llr2d) < 1e-10)
