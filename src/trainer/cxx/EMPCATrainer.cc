/**
 * @file trainer/cxx/EMPCATrainer.cc
 * @date Tue Oct 11 12:18:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Probabilistic Principal Component Analysis implemented using
 * Expectation Maximization. Implementation.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "bob/trainer/EMPCATrainer.h"
#include "bob/io/Exception.h"
#include "bob/core/array_copy.h"
#include "bob/core/array_type.h"
#include "bob/math/linear.h"
#include "bob/math/det.h"
#include "bob/math/inv.h"
#include "bob/math/stats.h"

namespace io = bob::io;
namespace mach = bob::machine;
namespace math = bob::math;
namespace train = bob::trainer;
namespace tca = bob::core::array;

train::EMPCATrainer::EMPCATrainer(int dimensionality, 
    double convergence_threshold, int max_iterations, bool compute_likelihood):
  EMTrainer<mach::LinearMachine, io::Arrayset>(convergence_threshold, 
  max_iterations, compute_likelihood), 
  m_dimensionality(dimensionality), m_S(0,0),
  m_z_first_order(0,dimensionality), 
  m_z_second_order(0,dimensionality,dimensionality),
  m_inW(dimensionality,dimensionality), m_invM(dimensionality,dimensionality),
  m_sigma2(0), m_f_log2pi(0), m_seed(-1),
  m_cache_dxf(0,0), m_cache_d(0), m_cache_f(0),
  m_cache_dxd_1(0,0), m_cache_dxd_2(0,0),
  m_cache_fxd_1(0,0), m_cache_fxd_2(0,0),
  m_cache_fxf_1(0,0), m_cache_fxf_2(0,0)
{
}

train::EMPCATrainer::EMPCATrainer(const train::EMPCATrainer& other):
  EMTrainer<mach::LinearMachine, io::Arrayset>(other.m_convergence_threshold, 
    other.m_max_iterations, other.m_compute_likelihood),
  m_dimensionality(other.m_dimensionality), 
  m_S(tca::ccopy(other.m_S)),
  m_z_first_order(tca::ccopy(other.m_z_first_order)), 
  m_z_second_order(tca::ccopy(other.m_z_second_order)), 
  m_inW(tca::ccopy(other.m_inW)),
  m_invM(tca::ccopy(other.m_invM)),
  m_sigma2(other.m_sigma2), m_f_log2pi(other.m_f_log2pi),
  m_seed(other.m_seed),
  m_cache_dxf(tca::ccopy(other.m_cache_dxf)),
  m_cache_d(tca::ccopy(other.m_cache_d)),
  m_cache_f(tca::ccopy(other.m_cache_f)),
  m_cache_dxd_1(tca::ccopy(other.m_cache_dxd_1)),
  m_cache_dxd_2(tca::ccopy(other.m_cache_dxd_2)),
  m_cache_fxd_1(tca::ccopy(other.m_cache_fxd_1)),
  m_cache_fxd_2(tca::ccopy(other.m_cache_fxd_2)),
  m_cache_fxf_1(tca::ccopy(other.m_cache_fxf_1)),
  m_cache_fxf_2(tca::ccopy(other.m_cache_fxf_2))
{
}

train::EMPCATrainer::~EMPCATrainer() {}

train::EMPCATrainer& train::EMPCATrainer::operator=
(const train::EMPCATrainer& other) 
{
  m_convergence_threshold = other.m_convergence_threshold;
  m_max_iterations = other.m_max_iterations;
  m_compute_likelihood = other.m_compute_likelihood;
  m_dimensionality = other.m_dimensionality;
  m_S = tca::ccopy(other.m_S);
  m_z_first_order = tca::ccopy(other.m_z_first_order);
  m_z_second_order = tca::ccopy(other.m_z_second_order);
  m_inW = tca::ccopy(other.m_inW);
  m_invM = tca::ccopy(other.m_invM);
  m_sigma2 = other.m_sigma2;
  m_f_log2pi = other.m_f_log2pi;
  m_seed = other.m_seed;
  m_cache_dxf = tca::ccopy(other.m_cache_dxf);
  m_cache_d = tca::ccopy(other.m_cache_d);
  m_cache_f = tca::ccopy(other.m_cache_f);
  m_cache_dxd_1 = tca::ccopy(other.m_cache_dxd_1);
  m_cache_dxd_2 = tca::ccopy(other.m_cache_dxd_2);
  m_cache_fxd_1 = tca::ccopy(other.m_cache_fxd_1);
  m_cache_fxd_2 = tca::ccopy(other.m_cache_fxd_2);
  m_cache_fxf_1 = tca::ccopy(other.m_cache_fxf_1);
  m_cache_fxf_2 = tca::ccopy(other.m_cache_fxf_2);
  return *this;
}

void train::EMPCATrainer::initialization(mach::LinearMachine& machine,
  const io::Arrayset& ar) 
{
  // checks for arrayset data type and shape once
  if(ar.getElementType() != bob::core::array::t_float64) {
    throw bob::io::TypeError(ar.getElementType(),
        bob::core::array::t_float64);
  }
  if(ar.getNDim() != 1) {
    throw bob::io::DimensionError(ar.getNDim(), 1);
  }

  // Gets dimension
  size_t n_features = ar.getShape()[0];

  // resizes the LinearMachine
  machine.resize(n_features, m_dimensionality); 

  // reinitializes array members
  initMembers(ar);

  // computes the mean and the covariance if required
  computeMeanVariance(machine, ar);

  // Random initialization of W and sigma2
  initRandomWSigma2(machine);

  // Computes the product m_inW = W^T.W
  computeWtW(machine);
  // Computes inverse(M), where M = Wt * W + sigma2 * Id
  computeInvM();
}

void train::EMPCATrainer::finalization(mach::LinearMachine& machine,
  const io::Arrayset& ar) 
{
}

void train::EMPCATrainer::initMembers(const io::Arrayset& ar) 
{
  // Gets dimensions
  size_t n_samples = ar.size();
  size_t n_features = ar.getShape()[0];

  // Covariance matrix S is only required to compute the log likelihood
  if(m_compute_likelihood)
    m_S.resize(n_features,n_features);
  else
    m_S.resize(0,0);
  m_z_first_order.resize(n_samples, m_dimensionality);  
  m_z_second_order.resize(n_samples, m_dimensionality, m_dimensionality);  
  m_inW.resize(m_dimensionality,m_dimensionality);
  m_invM.resize(m_dimensionality,m_dimensionality);
  m_sigma2 = 0.;
  m_f_log2pi = n_features * log(2*M_PI);

  // Cache
  m_cache_dxf.resize(m_dimensionality,n_features);
  m_cache_d.resize(m_dimensionality);
  m_cache_f.resize(n_features);
  m_cache_dxd_1.resize(m_dimensionality,m_dimensionality);
  m_cache_dxd_2.resize(m_dimensionality,m_dimensionality);
  m_cache_fxd_1.resize(n_features,m_dimensionality);
  m_cache_fxd_2.resize(n_features,m_dimensionality);
  // The following large cache matrices are only required to compute the 
  // log likelihood.
  if(m_compute_likelihood) 
  { 
    m_cache_fxf_1.resize(n_features,n_features);
    m_cache_fxf_2.resize(n_features,n_features);
  }
  else 
  {
    m_cache_fxf_1.resize(0,0);
    m_cache_fxf_2.resize(0,0);
  }
}

void train::EMPCATrainer::computeMeanVariance(mach::LinearMachine& machine, 
  const io::Arrayset& ar) 
{
  size_t n_samples = ar.size();
  size_t n_features = ar.getShape()[0];
  blitz::Array<double,1> mu = machine.updateInputDivision();
  if(m_compute_likelihood) 
  {
    // loads all the data in a single shot - required for scatter
    blitz::Array<double,2> data(n_features, n_samples);
    blitz::Range all = blitz::Range::all();
    for (size_t i=0; i<n_samples; ++i)
      data(all,i) = ar.get<double,1>(i);
    // Mean and scatter computation
    math::scatter(data, m_S, mu);
    // divides scatter by N-1
    m_S /= static_cast<double>(n_samples-1);
  }
  else 
  {
    // computes the mean and updates mu
    mu = 0.;
    for (size_t i=0; i<n_samples; ++i)
      mu += ar.get<double,1>(i);
    mu /= static_cast<double>(n_samples);
  }
}

void train::EMPCATrainer::initRandomWSigma2(mach::LinearMachine& machine) 
{
  // Initializes the random number generator
  boost::mt19937 rng;
  if(m_seed != -1)
    rng.seed((uint32_t)m_seed);
  boost::uniform_01<> range01;
  boost::variate_generator<boost::mt19937&, boost::uniform_01<> > die(rng, range01);
    
  // W initialization
  blitz::Array<double,2> W = machine.updateWeights();
  double ratio = 2.; /// Follows matlab implementation using a ratio of 2
  for(int j=0; j<W.extent(0); ++j)
    for(int i=0; i<W.extent(1); ++i)
      W(j,i) = die() * ratio;
  // sigma2 initialization
  m_sigma2 = die() * ratio;
}

void train::EMPCATrainer::computeWtW(mach::LinearMachine& machine) 
{
  blitz::Array<double,2> W = machine.updateWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0);
  math::prod(Wt, W, m_inW);
}

void train::EMPCATrainer::computeInvM() 
{
  // Compute inverse(M), where M = W^T * W + sigma2 * Id
  bob::math::eye(m_cache_dxd_1); // m_cache_dxd_1 = Id
  m_cache_dxd_1 *= m_sigma2; // m_cache_dxd_1 = sigma2 * Id
  m_cache_dxd_1 += m_inW; // m_cache_dxd_1 = M = W^T * W + sigma2 * Id
  bob::math::inv(m_cache_dxd_1, m_invM); // m_invM = inv(M)  
}
 


void train::EMPCATrainer::eStep(mach::LinearMachine& machine, const io::Arrayset& ar) 
{  
  // Gets mu and W from the machine
  const blitz::Array<double,1>& mu = machine.getInputDivision();
  blitz::Array<double,2> W = machine.updateWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0); // W^T

  // Computes the statistics
  for(size_t i=0; i<ar.size(); ++i)
  {
    /// 1/ First order statistics: z_first_order_i = inv(M) * W^T * (t - mu)
    // m_cache_f = t (sample) - mu (normalized sample)
    m_cache_f = ar.get<double,1>(i) - mu;
    // m_cache_dxf = inv(M) * W^T
    bob::math::prod(m_invM, Wt, m_cache_dxf);
    blitz::Array<double,1> z_first_order_i = m_z_first_order(i,blitz::Range::all());
    // z_first_order_i = inv(M) * W^T * (t - mu)
    bob::math::prod(m_cache_dxf, m_cache_f, z_first_order_i);

    /// 2/ Second order statistics: 
    ///     z_second_order_i = sigma2 * inv(M) + z_first_order_i * z_first_order_i^T
    blitz::Array<double,2> z_second_order_i = m_z_second_order(i,blitz::Range::all(),blitz::Range::all());
    // m_cache_dxd = z_first_order_i * z_first_order_i^T
    bob::math::prod(z_first_order_i, z_first_order_i, m_cache_dxd_1); // outer product
    // z_second_order_i = sigma2 * inv(M)
    z_second_order_i = m_invM;
    z_second_order_i *= m_sigma2;
    // z_second_order_i = sigma2 * inv(M) + z_first_order_i * z_first_order_i^T
    z_second_order_i += m_cache_dxd_1;
  }
}

void train::EMPCATrainer::mStep(mach::LinearMachine& machine, const io::Arrayset& ar) 
{
  // 1/ New estimate of W
  updateW(machine, ar);

  // 2/ New estimate of sigma2
  updateSigma2(machine, ar);

  // Computes the new value of inverse(M), where M = Wt * W + sigma2 * Id
  computeInvM();
}

void train::EMPCATrainer::updateW(mach::LinearMachine& machine, const io::Arrayset& ar) {
  // Get the mean mu and the projection matrix W
  const blitz::Array<double,1>& mu = machine.getInputDivision();
  blitz::Array<double,2>& W = machine.updateWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0); // W^T

  // Compute W = sum{ (t_{i} - mu) z_first_order_i^T} * inv( sum{z_second_order_i} )
  m_cache_fxd_1 = 0.;
  m_cache_dxd_1 = 0.;
  for(size_t i=0; i<ar.size(); ++i)
  {
    // m_cache_f = t (sample) - mu (normalized sample)
    m_cache_f = ar.get<double,1>(i) - mu;
    // first order statistics of sample i
    blitz::Array<double,1> z_first_order_i = m_z_first_order(i,blitz::Range::all());
    // m_cache_fxd_2 = (t - mu)*z_first_order_i
    bob::math::prod(m_cache_f, z_first_order_i, m_cache_fxd_2);
    m_cache_fxd_1 += m_cache_fxd_2;

    // second order statistics of sample i
    blitz::Array<double,2> z_second_order_i = m_z_second_order(i,blitz::Range::all(),blitz::Range::all());
    m_cache_dxd_1 += z_second_order_i;
  }

  // m_cache_dxd_2 = inv( sum(E(x_i.x_i^T)) )
  bob::math::inv(m_cache_dxd_1, m_cache_dxd_2);
  // New estimates of W
  bob::math::prod(m_cache_fxd_1, m_cache_dxd_2, W);
  // Updates W'*W as well
  math::prod(Wt, W, m_inW);
}

void train::EMPCATrainer::updateSigma2(mach::LinearMachine& machine, const io::Arrayset& ar) {
  // Get the mean mu and the projection matrix W
  const blitz::Array<double,1>& mu = machine.getInputDivision();
  blitz::Array<double,2>& W = machine.updateWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0); // W^T

  m_sigma2 = 0.;
  for(size_t i=0; i<ar.size(); ++i)
  {
    // a. sigma2 += || t - mu ||^2
    // m_cache_f = t (sample) - mu (normalized sample)
    m_cache_f = ar.get<double,1>(i) - mu;
    // sigma2 += || t - mu ||^2
    m_sigma2 += blitz::sum(blitz::pow2(m_cache_f));

    // b. sigma2 -= 2*E(x_i)^T*W^T*(t - mu)
    // m_cache_d = W^T*(t - mu)
    bob::math::prod(Wt, m_cache_f, m_cache_d);
    // first order of i
    blitz::Array<double,1> z_first_order_i = m_z_first_order(i,blitz::Range::all());
    // sigma2 -= 2*E(x_i)^T*W^T*(t - mu)
    m_sigma2 -= 2*bob::math::dot(z_first_order_i, m_cache_d);

    // c. sigma2 += trace( E(x_i.x_i^T)*W^T*W )
    // second order of i
    blitz::Array<double,2> z_second_order_i = m_z_second_order(i,blitz::Range::all(),blitz::Range::all());
    // m_cache_dxd_1 = E(x_i.x_i^T)*W^T*W
    bob::math::prod(z_second_order_i, m_inW, m_cache_dxd_1);
    // sigma2 += trace( E(x_i.x_i^T)*W^T*W )
    m_sigma2 += bob::math::trace(m_cache_dxd_1);
  }
  // Normalization factor
  m_sigma2 /= (static_cast<double>(ar.size()) * mu.extent(0));
}

double train::EMPCATrainer::computeLikelihood(mach::LinearMachine& machine)
{
  // Get W projection matrix
  blitz::Array<double,2>& W = machine.updateWeights();
  const blitz::Array<double,2> Wt = W.transpose(1,0); // W^T
  size_t n_features = m_S.extent(0);

  // 1/ Compute det(C), where C = sigma2.I + W.W^T
  //            det(C) = det(sigma2 * C / sigma2) = det(sigma2 * Id) * det(C / sigma2)
  //    We are using Sylvester's determinant theorem to compute a dxd 
  //    determinant rather than a fxf one: det(I + A.B) = det(I + B.A)
  //            det(C) = sigma2^n_features * det(I + W.W^T/sigma2)
  //                   = sigma2^n_features * det(I + W^T.W/sigma2) (cf. Bishop Appendix C)
  // detC = det( eye(n_features) * sigma2 )

  // detC = sigma2^n_features 
  double detC = pow(m_sigma2, n_features);
  // m_cache_dxd_1 = Id
  bob::math::eye(m_cache_dxd_1);
  // m_cache_dxd_2 = W^T.W
  bob::math::prod(Wt, W, m_cache_dxd_2);
  // m_cache_dxd_2 = W^T.W / sigma2
  m_cache_dxd_2 /= m_sigma2;
  // m_cache_dxd_1 = Id + W^T.W / sigma2
  m_cache_dxd_1 += m_cache_dxd_2;
  // detC = sigma2^n_features * det(I + W^T.W/sigma2)
  detC *= bob::math::det(m_cache_dxd_1);

  // 2/ Compute inv(C), where C = sigma2.I + W.W^T
  //    We are using the following identity (Property C.7 of Bishop's book)
  //      (A + B.D^-1.C)^-1 = A^-1 - A^-1.B(D+C.A^-1.B)^-1.C.A^-1
  //    Hence, inv(C) = sigma2^-1 .(I - W.M^-1.W^T)

  // Compute inverse(M), where M = Wt * W + sigma2 * Id
  computeInvM();
  // m_cache_fxf_1 = I = eye(n_features) 
  bob::math::eye(m_cache_fxf_1);
  // m_cache_fxd_1 = W * inv(M)
  bob::math::prod(W, m_invM, m_cache_fxd_1);
  // m_cache_fxf_2 = (W * inv(M) * Wt)
  bob::math::prod(m_cache_fxd_1, Wt, m_cache_fxf_2);
  // m_cache_fxd_1 = inv(C) = (I - W.M^-1.W^T) / sigma2
  m_cache_fxf_1 -= m_cache_fxf_2;
  m_cache_fxf_1 /= m_sigma2;

  // 3/ Compute inv(C).S
  bob::math::prod(m_cache_fxf_1, m_S, m_cache_fxf_2);

  // 4/ Use previous values to compute the log likelihood:
  // Log likelihood =  - N/2*{ d*ln(2*PI) + ln |detC| + tr(C^-1.S) }
  double llh = - static_cast<double>(m_z_first_order.extent(0)) / 2. * 
    ( m_f_log2pi + log(fabs(detC)) + bob::math::trace(m_cache_fxf_2) ); 

  return llh;
}
