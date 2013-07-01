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

import sys, unittest
import bob
import numpy, numpy.linalg

class PythonPLDATrainer():
  """A simplified (and slower) version of the PLDATrainer"""

  def __init__(self, convergence_threshold=0.001, max_iterations=10, 
      compute_likelihood=False, use_sum_second_order=True):
    # Our state
    self.m_convergence_threshold = convergence_threshold
    self.m_max_iterations = max_iterations
    self.m_compute_likelihood = compute_likelihood
    self.m_dim_f = 0
    self.m_dim_g = 0
    self.m_B = numpy.ndarray(shape=(0,0), dtype=numpy.float64)
    self.m_n_samples_per_id = numpy.ndarray(shape=(0,), dtype=numpy.float64)
    self.m_z_first_order = []
    self.m_z_second_order = []
    self.m_sum_z_second_order = numpy.ndarray(shape=(0,0), dtype=numpy.float64)

  def reset(self):
    """Resets our internal state"""
    self.m_convergence_threshold = 0.001
    self.m_max_iterations = 10
    self.m_compute_likelihood = False
    self.m_dim_f = 0
    self.m_dim_g = 0
    self.m_n_samples_per_id = numpy.ndarray(shape=(0,), dtype=numpy.float64)
    self.m_z_first_order = []
    self.m_z_second_order = []
    self.m_sum_z_second_order = numpy.ndarray(shape=(0,0), dtype=numpy.float64)

  def __check_training_data__(self, data):
    if len(data) == 0:
      raise RuntimeError("Training data set is empty")
    n_features = data[0].shape[1]
    for v in data:
      if(v.shape[1] != n_features):
        raise RuntimeError("Inconsistent feature dimensionality in training data set")

  def __init_members__(self, data):
    n_features = data[0].shape[1]
    self.m_z_first_order = []
    df_dg = self.m_dim_f+self.m_dim_g
    self.m_sum_z_second_order.resize(df_dg, df_dg)
    self.m_n_samples_per_id.resize(len(data))
    self.m_B.resize(n_features, df_dg)
    for i in range(len(data)):
      ns_i = data[i].shape[0]
      self.m_n_samples_per_id[i] = ns_i
      self.m_z_first_order.append(numpy.ndarray(shape=(ns_i, df_dg), dtype=numpy.float64))
      self.m_z_second_order.append(numpy.ndarray(shape=(ns_i, df_dg, df_dg), dtype=numpy.float64))

  def __init_mu__(self, machine, data):
    mu = numpy.zeros(shape=machine.mu.shape[0], dtype=numpy.float64)
    c = 0
    # Computes the mean of the data
    for v in data:
      for i in range(v.shape[0]):
        mu += v[i,:]
        c +=1
    mu /= c
    machine.mu = mu
 
  def __init_f__(self, machine, data):
    n_ids = len(data)
    S = numpy.zeros(shape=(machine.dim_d, n_ids), dtype=numpy.float64)
    Si_sum = numpy.zeros(shape=(machine.dim_d,), dtype=numpy.float64)
    for i in range(n_ids):
      Si = S[:,i]
      data_i = data[i]
      for j in range(data_i.shape[0]):
        Si += data_i[j,:]
      Si /= data_i.shape[0]
      Si_sum += Si
    Si_sum /= n_ids

    S = S - numpy.tile(Si_sum.reshape([machine.dim_d,1]), [1,n_ids])
    U, sigma, S_ = numpy.linalg.svd(S, full_matrices=False)
    U_slice = U[:,0:self.m_dim_f]
    sigma_slice = sigma[0:self.m_dim_f]
    sigma_slice_sqrt = numpy.sqrt(sigma_slice)
    machine.f = U_slice / sigma_slice_sqrt

  def __init_g__(self, machine, data):
    n_samples = 0
    for v in data:
      n_samples += v.shape[0]
    S = numpy.zeros(shape=(machine.dim_d, n_samples), dtype=numpy.float64)
    Si_sum = numpy.zeros(shape=(machine.dim_d,), dtype=numpy.float64)
    cache = numpy.zeros(shape=(machine.dim_d,), dtype=numpy.float64)
    c = 0
    for i in range(len(data)):
      cache = 0
      data_i = data[i]
      for j in range(data_i.shape[0]):
        cache += data_i[j,:]
      cache /= data_i.shape[0]
      for j in range(data_i.shape[0]):
        S[:,c] = data_i[j,:] - cache
        Si_sum += S[:,c]
        c += 1
    Si_sum /= n_samples

    S = S - numpy.tile(Si_sum.reshape([machine.dim_d,1]), [1,n_samples])
    U, sigma, S_ = numpy.linalg.svd(S, full_matrices=False)
    U_slice = U[:,0:self.m_dim_g]
    sigma_slice_sqrt = numpy.sqrt(sigma[0:self.m_dim_g])
    machine.g = U_slice / sigma_slice_sqrt

  def __init_sigma__(self, machine, data, factor = 1.):
    """As a variance of the data""" 
    cache1 = numpy.zeros(shape=(machine.dim_d,), dtype=numpy.float64)
    cache2 = numpy.zeros(shape=(machine.dim_d,), dtype=numpy.float64)
    n_samples = 0
    for v in data:
      for j in range(v.shape[0]):
        cache1 += v[j,:]
      n_samples += v.shape[0]
    cache1 /= n_samples
    for v in data:
      for j in range(v.shape[0]):
        cache2 += numpy.square(v[j,:] - cache1)
    machine.sigma = factor * cache2 / (n_samples - 1)

  def __init_mu_f_g_sigma__(self, machine, data):
    self.__init_mu__(machine, data)
    self.__init_f__(machine, data)
    self.__init_g__(machine, data)
    self.__init_sigma__(machine, data)

  def initialize(self, machine, data):
    self.__check_training_data__(data)
    n_features = data[0].shape[1]
    if(machine.dim_d != n_features):
      raise RuntimeError("Inconsistent feature dimensionality between the machine and the training data set")
    self.m_dim_f = machine.dim_f
    self.m_dim_g = machine.dim_g
    self.__init_members__(data)
    # Warning: Default initialization of mu, F, G, sigma using scatters
    self.__init_mu_f_g_sigma__(machine, data)
    # Make sure that the precomputation has been performed
    machine.__precompute__()

  def __compute_sufficient_statistics_given_observations__(self, machine, observations):
    """
    We compute the expected values of the latent variables given the observations 
    and parameters of the model.
    
    First order or the expected value of the latent variables.:
      F = (I+A^{T}\Sigma'^{-1}A)^{-1} * A^{T}\Sigma^{-1} (\tilde{x}_{s}-\mu').
    Second order stats:
      S = (I+A^{T}\Sigma'^{-1}A)^{-1} + (F*F^{T}).
    """

    # Get the number of observations 
    J_i                       = observations.shape[0]            # An integer > 0
    dim_d                     = observations.shape[1]            # A scalar
    # Useful values
    mu                        = machine.mu
    F                         = machine.f
    G                         = machine.g
    sigma                     = machine.sigma
    isigma                    = machine.__isigma__
    alpha                     = machine.__alpha__
    ft_beta                   = machine.__ft_beta__
    gamma                     = machine.get_add_gamma(J_i)
    # Normalise the observations
    normalised_observations   = observations - numpy.tile(mu, [J_i,1]) # (dim_d, J_i)

    ### Expected value of the latent variables using the scalable solution
    # Identity part first
    sum_ft_beta_part          = numpy.zeros(self.m_dim_f)     # (dim_f)
    for j in range(0, J_i):
      current_observation     = normalised_observations[j,:]  # (dim_d)
      sum_ft_beta_part        = sum_ft_beta_part + numpy.dot(ft_beta, current_observation)  # (dim_f)
    h_i                       = numpy.dot(gamma, sum_ft_beta_part)                          # (dim_f)
    # Reproject the identity part to work out the session parts
    Fh_i                      = numpy.dot(F, h_i)                                           # (dim_d)
    z_first_order = numpy.zeros((J_i, self.m_dim_f+self.m_dim_g))
    for j in range(0, J_i):
      current_observation       = normalised_observations[j,:]                  # (dim_d)
      w_ij                      = numpy.dot(alpha, G.transpose())               # (dim_g, dim_d)
      w_ij                      = numpy.multiply(w_ij, isigma)                  # (dim_g, dim_d)
      w_ij                      = numpy.dot(w_ij, (current_observation - Fh_i)) # (dim_g)
      z_first_order[j,:]        = numpy.hstack([h_i,w_ij])                      # (dim_f+dim_g)

    ### Calculate the expected value of the squared of the latent variables
    # The constant matrix we use has the following parts: [top_left, top_right; bottom_left, bottom_right]
    # P             = Inverse_I_plus_GTEG * G^T * Sigma^{-1} * F  (dim_g, dim_f)
    # top_left      = gamma                                       (dim_f, dim_f)
    # bottom_left   = top_right^T = P * gamma                     (dim_g, dim_f)
    # bottom_right  = Inverse_I_plus_GTEG - bottom_left * P^T     (dim_g, dim_g)
    top_left                 = gamma
    P                        = numpy.dot(alpha, G.transpose())
    P                        = numpy.dot(numpy.dot(P,numpy.diag(isigma)), F)
    bottom_left              = -1 * numpy.dot(P, top_left)
    top_right                = bottom_left.transpose()
    bottom_right             = alpha -1 * numpy.dot(bottom_left, P.transpose())
    constant_matrix          = numpy.bmat([[top_left,top_right],[bottom_left, bottom_right]])

    # Now get the actual expected value
    z_second_order = numpy.zeros((J_i, self.m_dim_f+self.m_dim_g, self.m_dim_f+self.m_dim_g))
    for j in range(0, J_i):
      z_second_order[j,:,:] = constant_matrix + numpy.outer(z_first_order[j,:],z_first_order[j,:])  # (dim_f+dim_g,dim_f+dim_g)

    ### Return the first and second order statistics
    return(z_first_order, z_second_order)

  def e_step(self, machine, data):
    self.m_sum_z_second_order.fill(0.)
    for i in range(len(data)):
      ### Get the observations for this label and the number of observations for this label.
      observations_for_h_i      = data[i]
      J_i                       = observations_for_h_i.shape[0]                           # An integer > 0
    
      ### Gather the statistics for this identity and then separate them for each observation.
      [z_first_order, z_second_order] = self.__compute_sufficient_statistics_given_observations__(machine, observations_for_h_i)
      self.m_z_first_order[i]  = z_first_order
      self.m_z_second_order[i] = z_second_order
      J_i = len(z_second_order)
      for j in range(0, J_i):
        self.m_sum_z_second_order += z_second_order[j]

  def __update_f_and_g__(self, machine, data):
    ### Initialise the numerator and the denominator.
    dim_d                          = machine.dim_d
    accumulated_B_numerator        = numpy.zeros((dim_d,self.m_dim_f+self.m_dim_g))
    accumulated_B_denominator      = numpy.linalg.inv(self.m_sum_z_second_order)
    mu                             = machine.mu

    ### Go through and process on a per subjectid basis
    for i in range(len(data)):
      # Normalise the observations
      J_i                       = data[i].shape[0]
      normalised_observations   = data[i] - numpy.tile(mu, [J_i,1]) # (J_i, dim_d)

      ### Gather the statistics for this label
      z_first_order_i                    = self.m_z_first_order[i]  # List of (dim_f+dim_g) vectors

      ### Accumulate for the B matrix for this identity (current_label).
      for j in range(0, J_i):
        current_observation_for_h_i   = normalised_observations[j,:]   # (dim_d)
        accumulated_B_numerator       = accumulated_B_numerator + numpy.outer(current_observation_for_h_i, z_first_order_i[j,:])  # (dim_d, dim_f+dim_g);

    ### Update the B matrix which we can then use this to update the F and G matrices.
    B                                  = numpy.dot(accumulated_B_numerator,accumulated_B_denominator)
    machine.f                          = B[:,0:self.m_dim_f].copy()
    machine.g                          = B[:,self.m_dim_f:self.m_dim_f+self.m_dim_g].copy()

  def __update_sigma__(self, machine, data):
    ### Initialise the accumulated Sigma
    dim_d                          = machine.dim_d
    mu                             = machine.mu
    accumulated_sigma              = numpy.zeros(dim_d)   # An array (dim_d)
    number_of_observations         = 0
    B = numpy.hstack([machine.f, machine.g])

    ### Go through and process on a per subjectid basis (based on the labels we were given.
    for i in range(len(data)):
      # Normalise the observations
      J_i                       = data[i].shape[0]
      normalised_observations   = data[i] - numpy.tile(mu, [J_i,1]) # (J_i, dim_d)

      ### Gather the statistics for this identity and then separate them for each
      ### observation.
      z_first_order_i                    = self.m_z_first_order[i]  # List of (dim_f+dim_g) vectors

      ### Accumulate for the sigma matrix, which will be diagonalised
      for j in range(0, J_i):
        current_observation_for_h_i   = normalised_observations[j,:]  # (dim_d)
        left                          = current_observation_for_h_i * current_observation_for_h_i # (dim_d)
        projected_direction           = numpy.dot(B, z_first_order_i[j,:])                        # (dim_d)
        right                         = projected_direction * current_observation_for_h_i         # (dim_d)
        accumulated_sigma             = accumulated_sigma + (left - right)                        # (dim_d)
        number_of_observations        = number_of_observations + 1

    ### Normalise by the number of observations (1/IJ)
    machine.sigma                     = accumulated_sigma / number_of_observations;

  def m_step(self, machine, data):
    self.__update_f_and_g__(machine, data)
    self.__update_sigma__(machine, data)
    machine.__precompute__()

  def finalize(self, machine, data):
    machine.__precompute_log_like__()

  def train(self, machine, data):
    self.initialize(machine, data)
    average_output_previous = -sys.maxint
    average_output = -sys.maxint
    self.e_step(machine, data)
    
    i = 0
    while True:
      average_output_previous = average_output
      self.m_step(machine, data)
      self.e_step(machine, data)
      if(self.m_max_iterations > 0 and i+1 >= self.m_max_iterations):
        break
      i += 1


class PLDATrainerTest(unittest.TestCase):
  """Performs various PLDA trainer tests."""
  
  def test01_plda_EM_vs_Python(self):

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

    # Runs the PLDA trainer EM-steps (2 steps)
    # Defines base trainer and machine
    t = bob.trainer.PLDATrainer(10)
    t_py = PythonPLDATrainer()
    m = bob.machine.PLDABase(D,nf,ng)
    m_py = bob.machine.PLDABase(D,nf,ng)

    # Sets the same initialization methods
    t.init_f_method = bob.trainer.PLDATrainer.BETWEEN_SCATTER
    t.init_g_method = bob.trainer.PLDATrainer.WITHIN_SCATTER
    t.init_sigma_method = bob.trainer.PLDATrainer.VARIANCE_DATA

    t.train(m, l)
    t_py.train(m_py, l)
    self.assertTrue(numpy.allclose(m.mu, m_py.mu))
    self.assertTrue(numpy.allclose(m.f, m_py.f))
    self.assertTrue(numpy.allclose(m.g, m_py.g))
    self.assertTrue(numpy.allclose(m.sigma, m_py.sigma))

  def test02_plda_EM_vs_Prince(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    dim_d = 7
    dim_f = 2
    dim_g = 3

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
      0.4364, -0.1699, -2.2015]).reshape(dim_d,dim_g)

    # F <-> PCA on G
    F_init=numpy.array([-0.054222647972093, -0.000000000783146, 
      0.596449127693018,  0.000000006265167, 
      0.298224563846509,  0.000000003132583, 
      0.447336845769764,  0.000000009397750, 
      -0.108445295944185, -0.000000001566292, 
      -0.501559493741856, -0.000000006265167, 
      -0.298224563846509, -0.000000003132583]).reshape(dim_d,dim_f)
    sigma_init = 0.01 * numpy.ones(dim_d, 'float64')

    # Defines reference results based on Princes'matlab implementation
    # After 1 iteration
    z_first_order_a_1 = numpy.array(
      [-2.624115900658397, -0.000000034277848,  1.554823055585319,  0.627476234024656, -0.264705934182394,
       -2.624115900658397, -0.000000034277848, -2.703482671599357, -1.533283607433197,  0.553725774828231,
       -2.624115900658397, -0.000000034277848,  2.311647528461115,  1.266362142140170, -0.317378177105131,
       -2.624115900658397, -0.000000034277848, -1.163402640008200, -0.372604542926019,  0.025152800097991
      ]).reshape(4, dim_f+dim_g)
    z_first_order_b_1 = numpy.array(
      [ 3.494168818797438,  0.000000045643026,  0.111295550530958, -0.029241422535725,  0.257045446451067,
        3.494168818797438,  0.000000045643026,  1.102110715965762,  1.481232954001794, -0.970661225144399,
        3.494168818797438,  0.000000045643026, -1.212854031699468, -1.435946529317718,  0.717884143973377
      ]).reshape(3, dim_f+dim_g)
  
    z_second_order_sum_1 = numpy.array(
      [64.203518285366087,  0.000000747228248,  0.002703277337642,  0.078542842475345,  0.020894328259862,
        0.000000747228248,  6.999999999999980, -0.000000003955962,  0.000000002017232, -0.000000003741593,
        0.002703277337642, -0.000000003955962, 19.136889380923918, 11.860493771107487, -4.584339465366988,
        0.078542842475345,  0.000000002017232, 11.860493771107487,  8.771502339750128, -3.905706024997424,
        0.020894328259862, -0.000000003741593, -4.584339465366988, -3.905706024997424,  2.011924970338584
      ]).reshape(dim_f+dim_g, dim_f+dim_g)

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
          -0.295956102084270, -0.000000006718366]).reshape(dim_d,dim_f)

    G_1 = numpy.array(
        [-1.836166150865047,  2.491475145758734,  5.095958946372235,
          -0.608732205531767, -0.618128420353493, -1.085423135463635,
          -0.697390472635929, -1.047900122276840, -6.080211153116984,
          0.769509301515319, -2.763610156675313, -5.972172587527176,
          1.332474692714491, -1.368103875407414, -2.096382536513033,
          0.304135903830416, -5.168096082564016, -9.604769461465978,
          0.597445549865284, -1.347101803379971, -5.900246013340080]).reshape(dim_d,dim_g)

    # After 2 iterations
    z_first_order_a_2 = numpy.array(
        [-2.144344161196005, -0.000000027851878,  1.217776189037369,  0.232492571855061, -0.212892893868819,
          -2.144344161196005, -0.000000027851878, -2.382647766948079, -1.759951013670071,  0.587213207926731,
          -2.144344161196005, -0.000000027851878,  2.143294830538722,  0.909307594408923, -0.183752098508072,
          -2.144344161196005, -0.000000027851878, -0.662558006326892,  0.717992497547010, -0.202897892977004
      ]).reshape(4, dim_f+dim_g)
    z_first_order_b_2 = numpy.array(
        [ 2.695117129662246,  0.000000035005543, -0.156173294945791, -0.123083763746364,  0.271123341933619,
          2.695117129662246,  0.000000035005543,  0.690321563509753,  0.944473716646212, -0.850835940962492,
          2.695117129662246,  0.000000035005543, -0.930970138998433, -0.949736472690315,  0.594216348861889
      ]).reshape(3, dim_f+dim_g)
 
    z_second_order_sum_2 = numpy.array(
        [41.602421167226410,  0.000000449434708, -1.513391506933811, -0.477818674270533,  0.059260102368316,
          0.000000449434708,  7.000000000000005, -0.000000023255959, -0.000000005157439, -0.000000003230262,
          -1.513391506933810, -0.000000023255959, 14.399631061987494,  8.068678077509025, -3.227586434905497,
          -0.477818674270533, -0.000000005157439,  8.068678077509025,  7.263248678863863, -3.060665688064639,
          0.059260102368316, -0.000000003230262, -3.227586434905497, -3.060665688064639,  1.705174220723198
      ]).reshape(dim_f+dim_g, dim_f+dim_g)

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
       -0.346593051621620, -0.000000007134046]).reshape(dim_d,dim_f)

    G_2 = numpy.array(
      [-2.266404374274820,  4.089199685832099,  7.023039382876370,
        0.094887459097613, -3.226829318470136, -3.452279917194724,
       -0.498398131733141, -1.651712333649899, -6.548008210704172,
        0.574932298590327, -2.198978667003715, -5.131253543126156,
        1.415857426810629, -1.627795701160212, -2.509013676007012,
       -0.543552834305580, -3.215063993186718, -7.006305082499653,
        0.562108137758111, -0.785296641855087, -5.318335345720314]).reshape(dim_d,dim_g)

    # Runs the PLDA trainer EM-steps (2 steps)
    
    # Defines base trainer and machine
    t = bob.trainer.PLDATrainer()
    t0 = bob.trainer.PLDATrainer(t)
    m = bob.machine.PLDABase(dim_d,dim_f,dim_g)
    t.initialize(m,l)
    m.sigma = sigma_init
    m.g = G_init
    m.f = F_init

    # Defines base trainer and machine (for Python implementation
    t_py = PythonPLDATrainer()
    m_py = bob.machine.PLDABase(dim_d,dim_f,dim_g)
    t_py.initialize(m_py,l)
    m_py.sigma = sigma_init
    m_py.g = G_init
    m_py.f = F_init
 
    # E-step 1
    t.e_step(m,l)
    t_py.e_step(m_py,l)
    # Compares statistics to Prince matlab reference
    self.assertTrue(numpy.allclose(t.z_first_order[0], z_first_order_a_1, 1e-10))
    self.assertTrue(numpy.allclose(t.z_first_order[1], z_first_order_b_1, 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order_sum, z_second_order_sum_1, 1e-10))
    # Compares statistics against the ones of the python implementation
    self.assertTrue(numpy.allclose(t.z_first_order[0], t_py.m_z_first_order[0], 1e-10))
    self.assertTrue(numpy.allclose(t.z_first_order[1], t_py.m_z_first_order[1], 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order_sum, t_py.m_sum_z_second_order, 1e-10))

    # M-step 1
    t.m_step(m,l)
    t_py.m_step(m_py,l)
    # Compares F, G and sigma to Prince matlab reference
    self.assertTrue(numpy.allclose(m.f, F_1, 1e-10))
    self.assertTrue(numpy.allclose(m.g, G_1, 1e-10))
    self.assertTrue(numpy.allclose(m.sigma, sigma_1, 1e-10))
    # Compares F, G and sigma to the ones of the python implementation
    self.assertTrue(numpy.allclose(m.f, m_py.f, 1e-10))
    self.assertTrue(numpy.allclose(m.g, m_py.g, 1e-10))
    self.assertTrue(numpy.allclose(m.sigma, m_py.sigma, 1e-10))

    # E-step 2
    t.e_step(m,l)
    t_py.e_step(m_py,l)
    # Compares statistics to Prince matlab reference
    self.assertTrue(numpy.allclose(t.z_first_order[0], z_first_order_a_2, 1e-10))
    self.assertTrue(numpy.allclose(t.z_first_order[1], z_first_order_b_2, 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order_sum, z_second_order_sum_2, 1e-10))
    # Compares statistics against the ones of the python implementation
    self.assertTrue(numpy.allclose(t.z_first_order[0], t_py.m_z_first_order[0], 1e-10))
    self.assertTrue(numpy.allclose(t.z_first_order[1], t_py.m_z_first_order[1], 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order_sum, t_py.m_sum_z_second_order, 1e-10))

    # M-step 2
    t.m_step(m,l)
    t_py.m_step(m_py,l)
    # Compares F, G and sigma to Prince matlab reference
    self.assertTrue(numpy.allclose(m.f, F_2, 1e-10))
    self.assertTrue(numpy.allclose(m.g, G_2, 1e-10))
    self.assertTrue(numpy.allclose(m.sigma, sigma_2, 1e-10))
    # Compares F, G and sigma to the ones of the python implementation
    self.assertTrue(numpy.allclose(m.f, m_py.f, 1e-10))
    self.assertTrue(numpy.allclose(m.g, m_py.g, 1e-10))
    self.assertTrue(numpy.allclose(m.sigma, m_py.sigma, 1e-10))


    # Test the second order statistics computation
    # Calls the initialization methods and resets randomly initialized values
    # to new reference ones (to make the tests deterministic)
    t.use_sum_second_order = False
    t.initialize(m,l)
    m.sigma = sigma_init
    m.g = G_init
    m.f = F_init
    t_py.initialize(m_py,l)
    m_py.sigma = sigma_init
    m_py.g = G_init
    m_py.f = F_init
 
    # E-step 1
    t.e_step(m,l)
    t_py.e_step(m_py,l)
    # Compares statistics to Prince matlab reference
    self.assertTrue(numpy.allclose(t.z_first_order[0], z_first_order_a_1, 1e-10))
    self.assertTrue(numpy.allclose(t.z_first_order[1], z_first_order_b_1, 1e-10))
    # Compares statistics against the ones of the python implementation
    self.assertTrue(numpy.allclose(t.z_first_order[0], t_py.m_z_first_order[0], 1e-10))
    self.assertTrue(numpy.allclose(t.z_first_order[1], t_py.m_z_first_order[1], 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order[0], t_py.m_z_second_order[0], 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order[1], t_py.m_z_second_order[1], 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order_sum, t_py.m_sum_z_second_order, 1e-10))

    # M-step 1
    t.m_step(m,l)
    t_py.m_step(m_py,l)
    # Compares F, G and sigma to the ones of the python implementation
    self.assertTrue(numpy.allclose(m.f, m_py.f, 1e-10))
    self.assertTrue(numpy.allclose(m.g, m_py.g, 1e-10))
    self.assertTrue(numpy.allclose(m.sigma, m_py.sigma, 1e-10))

    # E-step 2
    t.e_step(m,l)
    t_py.e_step(m_py,l)
    # Compares statistics to Prince matlab reference
    self.assertTrue(numpy.allclose(t.z_first_order[0], z_first_order_a_2, 1e-10))
    self.assertTrue(numpy.allclose(t.z_first_order[1], z_first_order_b_2, 1e-10))
    # Compares statistics against the ones of the python implementation
    self.assertTrue(numpy.allclose(t.z_first_order[0], t_py.m_z_first_order[0], 1e-10))
    self.assertTrue(numpy.allclose(t.z_first_order[1], t_py.m_z_first_order[1], 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order[0], t_py.m_z_second_order[0], 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order[1], t_py.m_z_second_order[1], 1e-10))
    self.assertTrue(numpy.allclose(t.z_second_order_sum, t_py.m_sum_z_second_order, 1e-10))

    # M-step 2
    t.m_step(m,l)
    t_py.m_step(m_py,l)
    # Compares F, G and sigma to the ones of the python implementation
    self.assertTrue(numpy.allclose(m.f, m_py.f, 1e-10))
    self.assertTrue(numpy.allclose(m.g, m_py.g, 1e-10))
    self.assertTrue(numpy.allclose(m.sigma, m_py.sigma, 1e-10))


  def test03_plda_enrollment(self):
    # Data used for performing the tests
    # Features and subspaces dimensionality
    dim_d = 7
    dim_f = 2
    dim_g = 3

    # initial values for F, G and sigma
    G_init=numpy.array([-1.1424, -0.5044, -0.1917,
      -0.6249,  0.1021, -0.8658,
      -1.1687,  1.1963,  0.1807,
      0.3926,  0.1203,  1.2665,
      1.3018, -1.0368, -0.2512,
      -0.5936, -0.8571, -0.2046,
      0.4364, -0.1699, -2.2015]).reshape(dim_d,dim_g)
    # F <-> PCA on G
    F_init=numpy.array([-0.054222647972093, -0.000000000783146, 
      0.596449127693018,  0.000000006265167, 
      0.298224563846509,  0.000000003132583, 
      0.447336845769764,  0.000000009397750, 
      -0.108445295944185, -0.000000001566292, 
      -0.501559493741856, -0.000000006265167, 
      -0.298224563846509, -0.000000003132583]).reshape(dim_d,dim_f)
    sigma_init = 0.01 * numpy.ones((dim_d,), 'float64')
    mean_zero = numpy.zeros((dim_d,), 'float64')

    # base machine
    mb = bob.machine.PLDABase(dim_d,dim_f,dim_g)
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
    t = bob.trainer.PLDATrainer()
    t.enrol(m, a_enrol)
    ll = m.compute_log_likelihood(x3)
    self.assertTrue(abs(ll - ll_ref) < 1e-10)

    # reference obtained by computing the likelihood of [x1,x2,x3], [x1,x2] 
    # and [x3] separately
    llr_ref = -4.43695386675
    llr = m.forward(x3)
    self.assertTrue(abs(llr - llr_ref) < 1e-10)
    #
    llr_separate = m.compute_log_likelihood(numpy.array([x1,x2,x3]), False) - \
      (m.compute_log_likelihood(numpy.array([x1,x2]), False) + m.compute_log_likelihood(numpy.array([x3]), False))
    self.assertTrue(abs(llr - llr_separate) < 1e-10)

  def test04_plda_comparisons(self):
    
    t1 = bob.trainer.PLDATrainer()
    t2 = bob.trainer.PLDATrainer()
    t2.rng = t1.rng
    self.assertTrue(  t1 == t2 )
    self.assertFalse( t1 != t2 )
    self.assertTrue(  t1.is_similar_to(t2) )

    training_set = [numpy.array([[1,2,3,4]], numpy.float64), numpy.array([[3,4,3,4]], numpy.float64)]
    m = bob.machine.PLDABase(4,1,1,1e-8)
    t1.rng.seed(37)
    t1.initialize(m, training_set)
    t1.e_step(m, training_set)
    t1.m_step(m, training_set)
    self.assertFalse( t1 == t2 )
    self.assertTrue(  t1 != t2 )
    self.assertFalse( t1.is_similar_to(t2) )
    t2.rng.seed(37)
    t2.initialize(m, training_set)
    t2.e_step(m, training_set)
    t2.m_step(m, training_set)
    self.assertTrue(  t1 == t2 )
    self.assertFalse( t1 != t2 )
    self.assertTrue(  t1.is_similar_to(t2) )
    t2.rng.seed(77)
    t2.initialize(m, training_set)
    t2.e_step(m, training_set)
    t2.m_step(m, training_set)
    self.assertFalse( t1 == t2 )
    self.assertTrue(  t1 != t2 )
    self.assertFalse( t1.is_similar_to(t2) )
