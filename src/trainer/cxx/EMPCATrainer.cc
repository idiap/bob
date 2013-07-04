/**
 * @file trainer/cxx/EMPCATrainer.cc
 * @date Tue Oct 11 12:18:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <algorithm>
#include <boost/random.hpp>
#include <cmath>

#include <bob/trainer/EMPCATrainer.h>
#include <bob/io/Exception.h>
#include <bob/core/array_copy.h>
#include <bob/core/array_type.h>
#include <bob/core/check.h>
#include <bob/machine/Exception.h>
#include <bob/math/linear.h>
#include <bob/math/det.h>
#include <bob/math/inv.h>
#include <bob/math/stats.h>

bob::trainer::EMPCATrainer::EMPCATrainer(double convergence_threshold,
    size_t max_iterations, bool compute_likelihood):
  EMTrainer<bob::machine::LinearMachine, blitz::Array<double,2> >(convergence_threshold, 
    max_iterations, compute_likelihood), 
  m_S(0,0),
  m_z_first_order(0,0), m_z_second_order(0,0,0),
  m_inW(0,0), m_invM(0,0), m_sigma2(0), m_f_log2pi(0), 
  m_tmp_dxf(0,0), m_tmp_d(0), m_tmp_f(0),
  m_tmp_dxd_1(0,0), m_tmp_dxd_2(0,0),
  m_tmp_fxd_1(0,0), m_tmp_fxd_2(0,0),
  m_tmp_fxf_1(0,0), m_tmp_fxf_2(0,0)
{
}

bob::trainer::EMPCATrainer::EMPCATrainer(const bob::trainer::EMPCATrainer& other):
  EMTrainer<bob::machine::LinearMachine, blitz::Array<double,2> >(other.m_convergence_threshold, 
    other.m_max_iterations, other.m_compute_likelihood),
  m_S(bob::core::array::ccopy(other.m_S)),
  m_z_first_order(bob::core::array::ccopy(other.m_z_first_order)), 
  m_z_second_order(bob::core::array::ccopy(other.m_z_second_order)), 
  m_inW(bob::core::array::ccopy(other.m_inW)),
  m_invM(bob::core::array::ccopy(other.m_invM)),
  m_sigma2(other.m_sigma2), m_f_log2pi(other.m_f_log2pi),
  m_tmp_dxf(bob::core::array::ccopy(other.m_tmp_dxf)),
  m_tmp_d(bob::core::array::ccopy(other.m_tmp_d)),
  m_tmp_f(bob::core::array::ccopy(other.m_tmp_f)),
  m_tmp_dxd_1(bob::core::array::ccopy(other.m_tmp_dxd_1)),
  m_tmp_dxd_2(bob::core::array::ccopy(other.m_tmp_dxd_2)),
  m_tmp_fxd_1(bob::core::array::ccopy(other.m_tmp_fxd_1)),
  m_tmp_fxd_2(bob::core::array::ccopy(other.m_tmp_fxd_2)),
  m_tmp_fxf_1(bob::core::array::ccopy(other.m_tmp_fxf_1)),
  m_tmp_fxf_2(bob::core::array::ccopy(other.m_tmp_fxf_2))
{
}

bob::trainer::EMPCATrainer::~EMPCATrainer()
{
}

bob::trainer::EMPCATrainer& bob::trainer::EMPCATrainer::operator=
  (const bob::trainer::EMPCATrainer& other) 
{
  if (this != &other)
  {
    bob::trainer::EMTrainer<bob::machine::LinearMachine,
      blitz::Array<double,2> >::operator=(other);
    m_S = bob::core::array::ccopy(other.m_S);
    m_z_first_order = bob::core::array::ccopy(other.m_z_first_order);
    m_z_second_order = bob::core::array::ccopy(other.m_z_second_order);
    m_inW = bob::core::array::ccopy(other.m_inW);
    m_invM = bob::core::array::ccopy(other.m_invM);
    m_sigma2 = other.m_sigma2;
    m_f_log2pi = other.m_f_log2pi;
    m_tmp_dxf = bob::core::array::ccopy(other.m_tmp_dxf);
    m_tmp_d = bob::core::array::ccopy(other.m_tmp_d);
    m_tmp_f = bob::core::array::ccopy(other.m_tmp_f);
    m_tmp_dxd_1 = bob::core::array::ccopy(other.m_tmp_dxd_1);
    m_tmp_dxd_2 = bob::core::array::ccopy(other.m_tmp_dxd_2);
    m_tmp_fxd_1 = bob::core::array::ccopy(other.m_tmp_fxd_1);
    m_tmp_fxd_2 = bob::core::array::ccopy(other.m_tmp_fxd_2);
    m_tmp_fxf_1 = bob::core::array::ccopy(other.m_tmp_fxf_1);
    m_tmp_fxf_2 = bob::core::array::ccopy(other.m_tmp_fxf_2);
  }
  return *this;
}

bool bob::trainer::EMPCATrainer::operator==
  (const bob::trainer::EMPCATrainer &other) const
{
  return bob::trainer::EMTrainer<bob::machine::LinearMachine,
           blitz::Array<double,2> >::operator==(other) &&
        bob::core::array::isEqual(m_S, other.m_S) &&
        bob::core::array::isEqual(m_z_first_order, other.m_z_first_order) &&
        bob::core::array::isEqual(m_z_second_order, other.m_z_second_order) &&
        bob::core::array::isEqual(m_inW, other.m_inW) &&
        bob::core::array::isEqual(m_invM, other.m_invM) &&
        m_sigma2 == other.m_sigma2 &&
        m_f_log2pi == other.m_f_log2pi;
}

bool bob::trainer::EMPCATrainer::operator!=
  (const bob::trainer::EMPCATrainer &other) const
{
  return !(this->operator==(other));
}

bool bob::trainer::EMPCATrainer::is_similar_to
  (const bob::trainer::EMPCATrainer &other, const double r_epsilon, 
   const double a_epsilon) const
{
  return bob::trainer::EMTrainer<bob::machine::LinearMachine,
           blitz::Array<double,2> >::is_similar_to(other, r_epsilon, a_epsilon) &&
        bob::core::array::isClose(m_S, other.m_S, r_epsilon, a_epsilon) &&
        bob::core::array::isClose(m_z_first_order, other.m_z_first_order, r_epsilon, a_epsilon) &&
        bob::core::array::isClose(m_z_second_order, other.m_z_second_order, r_epsilon, a_epsilon) &&
        bob::core::array::isClose(m_inW, other.m_inW, r_epsilon, a_epsilon) &&
        bob::core::array::isClose(m_invM, other.m_invM, r_epsilon, a_epsilon) &&
        bob::core::isClose(m_sigma2, other.m_sigma2, r_epsilon, a_epsilon) &&
        bob::core::isClose(m_f_log2pi, other.m_f_log2pi, r_epsilon, a_epsilon);
}

void bob::trainer::EMPCATrainer::initialize(bob::machine::LinearMachine& machine,
  const blitz::Array<double,2>& ar) 
{
  // reinitializes array members and checks dimensionality
  initMembers(machine, ar);

  // computes the mean and the covariance if required
  computeMeanVariance(machine, ar);

  // Random initialization of W and sigma2
  initRandomWSigma2(machine);

  // Computes the product m_inW = W^T.W
  computeWtW(machine);
  // Computes inverse(M), where M = Wt * W + sigma2 * Id
  computeInvM();
}

void bob::trainer::EMPCATrainer::finalize(bob::machine::LinearMachine& machine,
  const blitz::Array<double,2>& ar) 
{
}

void bob::trainer::EMPCATrainer::initMembers(
  const bob::machine::LinearMachine& machine,
  const blitz::Array<double,2>& ar)
{
  // Gets dimensions
  const size_t n_samples = ar.extent(0);
  const size_t n_features = ar.extent(1);

  // Checks that the dimensions are matching 
  const size_t n_inputs = machine.inputSize();
  const size_t n_outputs = machine.outputSize();

  // Checks that the dimensions are matching
  if (n_inputs != n_features)
    throw bob::machine::NInputsMismatch(n_inputs, n_features);

  // Covariance matrix S is only required to compute the log likelihood
  if (m_compute_likelihood)
    m_S.resize(n_features,n_features);
  else
    m_S.resize(0,0);
  m_z_first_order.resize(n_samples, n_outputs);  
  m_z_second_order.resize(n_samples, n_outputs, n_outputs);
  m_inW.resize(n_outputs, n_outputs);
  m_invM.resize(n_outputs, n_outputs);
  m_sigma2 = 0.;
  m_f_log2pi = n_features * log(2*M_PI);

  // Cache
  m_tmp_dxf.resize(n_outputs, n_features);
  m_tmp_d.resize(n_outputs);
  m_tmp_f.resize(n_features);
  m_tmp_dxd_1.resize(n_outputs, n_outputs);
  m_tmp_dxd_2.resize(n_outputs, n_outputs);
  m_tmp_fxd_1.resize(n_features, n_outputs);
  m_tmp_fxd_2.resize(n_features, n_outputs);
  // The following large cache matrices are only required to compute the 
  // log likelihood.
  if (m_compute_likelihood) 
  { 
    m_tmp_fxf_1.resize(n_features, n_features);
    m_tmp_fxf_2.resize(n_features, n_features);
  }
  else 
  {
    m_tmp_fxf_1.resize(0,0);
    m_tmp_fxf_2.resize(0,0);
  }
}

void bob::trainer::EMPCATrainer::computeMeanVariance(bob::machine::LinearMachine& machine, 
  const blitz::Array<double,2>& ar) 
{
  size_t n_samples = ar.extent(0);
  size_t n_features = ar.extent(1);
  blitz::Array<double,1> mu = machine.updateInputSubtraction();
  blitz::Range all = blitz::Range::all();
  if (m_compute_likelihood) 
  {
    // Mean and scatter computation
    bob::math::scatter(ar, m_S, mu);
    // divides scatter by N-1
    m_S /= static_cast<double>(n_samples-1);
  }
  else 
  {
    // computes the mean and updates mu
    mu = 0.;
    for (size_t i=0; i<n_samples; ++i)
      mu += ar(i,all);
    mu /= static_cast<double>(n_samples);
  }
}

void bob::trainer::EMPCATrainer::initRandomWSigma2(bob::machine::LinearMachine& machine) 
{
  // Initializes the random number generator
  boost::uniform_01<> range01;
  boost::variate_generator<boost::mt19937&, boost::uniform_01<> > die(*m_rng, range01);
    
  // W initialization (TODO: add method in core)
  blitz::Array<double,2> W = machine.updateWeights();
  double ratio = 2.; /// Follows matlab implementation using a ratio of 2
  for (int i=0; i<W.extent(0); ++i)
    for (int j=0; j<W.extent(1); ++j)
      W(i,j) = die() * ratio;
  // sigma2 initialization
  m_sigma2 = die() * ratio;
}

void bob::trainer::EMPCATrainer::computeWtW(bob::machine::LinearMachine& machine) 
{
  const blitz::Array<double,2> W = machine.getWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0);
  bob::math::prod(Wt, W, m_inW);
}

void bob::trainer::EMPCATrainer::computeInvM() 
{
  // Compute inverse(M), where M = W^T * W + sigma2 * Id
  bob::math::eye(m_tmp_dxd_1); // m_tmp_dxd_1 = Id
  m_tmp_dxd_1 *= m_sigma2; // m_tmp_dxd_1 = sigma2 * Id
  m_tmp_dxd_1 += m_inW; // m_tmp_dxd_1 = M = W^T * W + sigma2 * Id
  bob::math::inv(m_tmp_dxd_1, m_invM); // m_invM = inv(M)  
}
 


void bob::trainer::EMPCATrainer::eStep(bob::machine::LinearMachine& machine, const blitz::Array<double,2>& ar) 
{  
  // Gets mu and W from the machine
  const blitz::Array<double,1>& mu = machine.getInputSubtraction();
  const blitz::Array<double,2>& W = machine.getWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0); // W^T

  // Computes the statistics
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<ar.extent(0); ++i)
  {
    /// 1/ First order statistics: \f$z_first_order_i = inv(M) W^T (t - \mu)\f$
    // m_tmp_f = t (sample) - mu (normalized sample)
    m_tmp_f = ar(i,a) - mu;
    // m_tmp_dxf = inv(M) * W^T
    bob::math::prod(m_invM, Wt, m_tmp_dxf);
    blitz::Array<double,1> z_first_order_i = m_z_first_order(i,blitz::Range::all());
    // z_first_order_i = inv(M) * W^T * (t - mu)
    bob::math::prod(m_tmp_dxf, m_tmp_f, z_first_order_i);

    /// 2/ Second order statistics: 
    ///     z_second_order_i = sigma2 * inv(M) + z_first_order_i * z_first_order_i^T
    blitz::Array<double,2> z_second_order_i = m_z_second_order(i,blitz::Range::all(),blitz::Range::all());
    // m_tmp_dxd = z_first_order_i * z_first_order_i^T
    bob::math::prod(z_first_order_i, z_first_order_i, m_tmp_dxd_1); // outer product
    // z_second_order_i = sigma2 * inv(M)
    z_second_order_i = m_invM;
    z_second_order_i *= m_sigma2;
    // z_second_order_i = sigma2 * inv(M) + z_first_order_i * z_first_order_i^T
    z_second_order_i += m_tmp_dxd_1;
  }
}

void bob::trainer::EMPCATrainer::mStep(bob::machine::LinearMachine& machine, const blitz::Array<double,2>& ar) 
{
  // 1/ New estimate of W
  updateW(machine, ar);

  // 2/ New estimate of sigma2
  updateSigma2(machine, ar);

  // Computes the new value of inverse(M), where M = Wt * W + sigma2 * Id
  computeInvM();
}

void bob::trainer::EMPCATrainer::updateW(bob::machine::LinearMachine& machine, const blitz::Array<double,2>& ar) {
  // Get the mean mu and the projection matrix W
  const blitz::Array<double,1>& mu = machine.getInputSubtraction();
  blitz::Array<double,2>& W = machine.updateWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0); // W^T

  // Compute W = sum{ (t_{i} - mu) z_first_order_i^T} * inv( sum{z_second_order_i} )
  m_tmp_fxd_1 = 0.;
  m_tmp_dxd_1 = 0.;
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<ar.extent(0); ++i)
  {
    // m_tmp_f = t (sample) - mu (normalized sample)
    m_tmp_f = ar(i,a) - mu;
    // first order statistics of sample i
    blitz::Array<double,1> z_first_order_i = m_z_first_order(i,blitz::Range::all());
    // m_tmp_fxd_2 = (t - mu)*z_first_order_i
    bob::math::prod(m_tmp_f, z_first_order_i, m_tmp_fxd_2);
    m_tmp_fxd_1 += m_tmp_fxd_2;

    // second order statistics of sample i
    blitz::Array<double,2> z_second_order_i = m_z_second_order(i,blitz::Range::all(),blitz::Range::all());
    m_tmp_dxd_1 += z_second_order_i;
  }

  // m_tmp_dxd_2 = inv( sum(E(x_i.x_i^T)) )
  bob::math::inv(m_tmp_dxd_1, m_tmp_dxd_2);
  // New estimates of W
  bob::math::prod(m_tmp_fxd_1, m_tmp_dxd_2, W);
  // Updates W'*W as well
  bob::math::prod(Wt, W, m_inW);
}

void bob::trainer::EMPCATrainer::updateSigma2(bob::machine::LinearMachine& machine, const blitz::Array<double,2>& ar) {
  // Get the mean mu and the projection matrix W
  const blitz::Array<double,1>& mu = machine.getInputSubtraction();
  const blitz::Array<double,2>& W = machine.getWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0); // W^T

  m_sigma2 = 0.;
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<ar.extent(0); ++i)
  {
    // a. sigma2 += || t - mu ||^2
    // m_tmp_f = t (sample) - mu (normalized sample)
    m_tmp_f = ar(i,a) - mu;
    // sigma2 += || t - mu ||^2
    m_sigma2 += blitz::sum(blitz::pow2(m_tmp_f));

    // b. sigma2 -= 2*E(x_i)^T*W^T*(t - mu)
    // m_tmp_d = W^T*(t - mu)
    bob::math::prod(Wt, m_tmp_f, m_tmp_d);
    // first order of i
    blitz::Array<double,1> z_first_order_i = m_z_first_order(i,blitz::Range::all());
    // sigma2 -= 2*E(x_i)^T*W^T*(t - mu)
    m_sigma2 -= 2*bob::math::dot(z_first_order_i, m_tmp_d);

    // c. sigma2 += trace( E(x_i.x_i^T)*W^T*W )
    // second order of i
    blitz::Array<double,2> z_second_order_i = m_z_second_order(i,blitz::Range::all(),blitz::Range::all());
    // m_tmp_dxd_1 = E(x_i.x_i^T)*W^T*W
    bob::math::prod(z_second_order_i, m_inW, m_tmp_dxd_1);
    // sigma2 += trace( E(x_i.x_i^T)*W^T*W )
    m_sigma2 += bob::math::trace(m_tmp_dxd_1);
  }
  // Normalization factor
  m_sigma2 /= (static_cast<double>(ar.extent(0)) * mu.extent(0));
}

double bob::trainer::EMPCATrainer::computeLikelihood(bob::machine::LinearMachine& machine)
{
  // Get W projection matrix
  const blitz::Array<double,2>& W = machine.getWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0); // W^T
  const size_t n_features = m_S.extent(0);

  // 1/ Compute det(C), where C = sigma2.I + W.W^T
  //            det(C) = det(sigma2 * C / sigma2) = det(sigma2 * Id) * det(C / sigma2)
  //    We are using Sylvester's determinant theorem to compute a dxd 
  //    determinant rather than a fxf one: det(I + A.B) = det(I + B.A)
  //            det(C) = sigma2^n_features * det(I + W.W^T/sigma2)
  //                   = sigma2^n_features * det(I + W^T.W/sigma2) (cf. Bishop Appendix C)
  // detC = det( eye(n_features) * sigma2 )

  // detC = sigma2^n_features 
  double detC = pow(m_sigma2, n_features);
  // m_tmp_dxd_1 = Id
  bob::math::eye(m_tmp_dxd_1);
  // m_tmp_dxd_2 = W^T.W
  bob::math::prod(Wt, W, m_tmp_dxd_2);
  // m_tmp_dxd_2 = W^T.W / sigma2
  m_tmp_dxd_2 /= m_sigma2;
  // m_tmp_dxd_1 = Id + W^T.W / sigma2
  m_tmp_dxd_1 += m_tmp_dxd_2;
  // detC = sigma2^n_features * det(I + W^T.W/sigma2)
  detC *= bob::math::det(m_tmp_dxd_1);

  // 2/ Compute inv(C), where C = sigma2.I + W.W^T
  //    We are using the following identity (Property C.7 of Bishop's book)
  //      (A + B.D^-1.C)^-1 = A^-1 - A^-1.B(D+C.A^-1.B)^-1.C.A^-1
  //    Hence, inv(C) = sigma2^-1 .(I - W.M^-1.W^T)

  // Compute inverse(M), where M = Wt * W + sigma2 * Id
  computeInvM();
  // m_tmp_fxf_1 = I = eye(n_features) 
  bob::math::eye(m_tmp_fxf_1);
  // m_tmp_fxd_1 = W * inv(M)
  bob::math::prod(W, m_invM, m_tmp_fxd_1);
  // m_tmp_fxf_2 = (W * inv(M) * Wt)
  bob::math::prod(m_tmp_fxd_1, Wt, m_tmp_fxf_2);
  // m_tmp_fxd_1 = inv(C) = (I - W.M^-1.W^T) / sigma2
  m_tmp_fxf_1 -= m_tmp_fxf_2;
  m_tmp_fxf_1 /= m_sigma2;

  // 3/ Compute inv(C).S
  bob::math::prod(m_tmp_fxf_1, m_S, m_tmp_fxf_2);

  // 4/ Use previous values to compute the log likelihood:
  // Log likelihood =  - N/2*{ d*ln(2*PI) + ln |detC| + tr(C^-1.S) }
  double llh = - static_cast<double>(m_z_first_order.extent(0)) / 2. * 
    ( m_f_log2pi + log(fabs(detC)) + bob::math::trace(m_tmp_fxf_2) ); 

  return llh;
}
