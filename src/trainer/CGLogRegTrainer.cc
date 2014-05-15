/**
 * @date Sat Sep 1 19:26:00 2012 +0100
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/trainer/CGLogRegTrainer.h>
#include <bob/math/linear.h>
#include <bob/core/logging.h>
#include <limits>

bob::trainer::CGLogRegTrainer::CGLogRegTrainer(const double prior,
  const double convergence_threshold, const size_t max_iterations,
  const double lambda, const bool mean_std_norm):
    m_prior(prior),
    m_convergence_threshold(convergence_threshold),
    m_max_iterations(max_iterations),
    m_lambda(lambda),
    m_mean_std_norm(mean_std_norm)
{
  if(prior<=0. || prior>=1.)
  {
    boost::format m("Prior (%f) not in the range ]0,1[.");
    m % prior;
    throw std::runtime_error(m.str());
  }
}

bob::trainer::CGLogRegTrainer::CGLogRegTrainer(const bob::trainer::CGLogRegTrainer& other):
  m_prior(other.m_prior),
  m_convergence_threshold(other.m_convergence_threshold),
  m_max_iterations(other.m_max_iterations),
  m_lambda(other.m_lambda),
  m_mean_std_norm(other.m_mean_std_norm)
{
}

bob::trainer::CGLogRegTrainer::~CGLogRegTrainer() {}

bob::trainer::CGLogRegTrainer& bob::trainer::CGLogRegTrainer::operator=
(const bob::trainer::CGLogRegTrainer& other)
{
  if(this != &other)
  {
    m_prior = other.m_prior;
    m_convergence_threshold = other.m_convergence_threshold;
    m_max_iterations = other.m_max_iterations;
    m_lambda = other.m_lambda;
    m_mean_std_norm = other.m_mean_std_norm;
  }
  return *this;
}

bool
bob::trainer::CGLogRegTrainer::operator==(const bob::trainer::CGLogRegTrainer& b) const
{
  return (this->m_prior == b.m_prior &&
          this->m_convergence_threshold == b.m_convergence_threshold &&
          this->m_max_iterations == b.m_max_iterations &&
          this->m_lambda == b.m_lambda &&
          this->m_mean_std_norm == b.m_mean_std_norm);
}

bool
bob::trainer::CGLogRegTrainer::operator!=(const bob::trainer::CGLogRegTrainer& b) const
{
  return !(this->operator==(b));
}

void bob::trainer::CGLogRegTrainer::train(bob::machine::LinearMachine& machine,
  const blitz::Array<double,2>& negatives, const blitz::Array<double,2>& positives) const
{
  // Checks for arraysets data type and shape once
  bob::core::array::assertSameDimensionLength(negatives.extent(1), positives.extent(1));

  // Data is checked now and conforms, just proceed w/o any further checks.
  size_t n_samples1 = positives.extent(0);
  size_t n_samples2 = negatives.extent(0);
  size_t n_samples = n_samples1 + n_samples2;
  size_t n_features = positives.extent(1);

  // Defines useful ranges
  blitz::Range rall = blitz::Range::all();
  blitz::Range rd = blitz::Range(0,n_features-1);
  blitz::Range r1 = blitz::Range(0,n_samples1-1);
  blitz::Range r2 = blitz::Range(n_samples1,n_samples-1);

  // indices for blitz iterations
  blitz::firstIndex i;
  blitz::secondIndex j;

  blitz::Array<double,1> mean(n_features);
  blitz::Array<double,1> std_dev(n_features);
  // mean and variance of the training data
  if (m_mean_std_norm){
    // collect all samples in one matrix
    blitz::Array<double,2> all_data(n_samples,n_features);
    for(size_t i=0; i<n_samples1; ++i)
      all_data(i,rall) = positives(i,rall);
    for(size_t i=0; i<n_samples2; ++i)
      all_data(i+n_samples1, rall) = negatives(i,rall);

    // compute mean and std-dev from samples
    mean = blitz::mean(all_data(j,i), j);
    blitz::Array<double,2> squared(blitz::pow2(all_data));
    std_dev = blitz::sqrt((blitz::sum(squared(j,i), j) - n_samples*blitz::pow2(mean)) / n_samples);
  } else {
    mean = 0.;
    std_dev = 1.;
  }

  // Creates a large blitz::Array containing the samples
  // x = |positives - negatives|, of size (n_features+1,n_samples1+n_samples2)
  //     |1.  -1. |
  blitz::Array<double,2> x(n_features+1, n_samples);
  x(n_features,r1) = 1.;
  x(n_features,r2) = -1.;
  for(size_t i=0; i<n_samples1; ++i)
    x(rd,i) = (positives(i,rall) - mean(rall)) / std_dev(rall);
  for(size_t i=0; i<n_samples2; ++i)
    x(rd,i+n_samples1) = -(negatives(i,rall) - mean(rall)) / std_dev(rall);

  // Ratio between the two classes and weights vector
  double prop = (double)n_samples1 / (double)n_samples;
  blitz::Array<double,1> weights(n_samples);
  weights(r1) = m_prior / prop;
  weights(r2) = (1.-m_prior) / (1.-prop);

  // Initializes offset vector
  blitz::Array<double,1> offset(n_samples);
  const double logit = log(m_prior/(1.-m_prior));
  offset(r1) = logit;
  offset(r2) = -logit;

  // Initializes gradient and w vectors
  blitz::Array<double,1> g_old(n_features+1);
  blitz::Array<double,1> w_old(n_features+1);
  blitz::Array<double,1> g(n_features+1);
  blitz::Array<double,1> w(n_features+1);
  g_old = 0.;
  w_old = 0.;
  g = 0.;
  w = 0.;

  // Initialize working arrays
  blitz::Array<double,1> s1(n_samples);
  blitz::Array<double,1> u(n_features+1);
  blitz::Array<double,1> tmp_n(n_samples);
  blitz::Array<double,1> tmp_d(n_features+1);

  // Iterates...
  static const double ten_epsilon = 10*std::numeric_limits<double>::epsilon();
  for(size_t iter=0; ; ++iter)
  {
    // 1. Computes the non-weighted version of the likelihood
    // s1 = sum_{i=1}^{n}(1./(1.+exp(-y_i (w^T x_i + logit))
    //   where - the x blitz::Array contains -y_i x_i values
    //         - the offset blitz::Array contains -y_i logit values
    s1 = 1. / (1. + blitz::exp(blitz::sum(w(j)*x(j,i), j) + offset));
    // 2. Likelihood weighted by the prior/proportion
    tmp_n = s1 * weights;
    // 3. Gradient g of this weighted likelihood wrt. the weight vector w
    bob::math::prod(x, tmp_n, g);
    g -= m_lambda * w; // Regularization

    // 4. Conjugate gradient step
    if(iter == 0)
      u = g;
    else
    {
      tmp_d = (g-g_old);
      double den = blitz::sum(u * tmp_d);
      if(den == 0)
        u = 0.;
      else
      {
        // Hestenes-Stiefel formula: Heuristic to set the scale factor beta
        //   (chosen as it works well in practice)
        // beta = g^t(g-g_old) / (u_old^T (g - g_old))
        double beta = blitz::sum(tmp_d * g) / den;
        u = g - beta * u;
      }
    }

    // 5. Line search along the direction u
    // a. Compute ux
    bob::math::prod(u,x,tmp_n);
    // b. Compute u^T H u
    //      = sum_{i} weights(i) sigmoid(w^T x_i) [1-sigmoid(w^T x_i)] (u^T x_i) + lambda u^T u
    double uhu = blitz::sum(blitz::pow2(tmp_n) * weights * s1 * (1.-s1)) + m_lambda*blitz::sum(blitz::pow2(u));
    // Terminates if uhu is close to zero
    if(fabs(uhu) < ten_epsilon)
    {
      bob::core::info << "# CGLogReg Training terminated: convergence after " << iter << " iterations (u^T H u == 0)." << std::endl;
      break;
    }
    // c. Compute w = w_old - (g^T u)/(u^T H u) u
    w = w + blitz::sum(u*g) / uhu * u;

    // Terminates if convergence has been reached
    if(blitz::max(blitz::fabs(w-w_old)) <= m_convergence_threshold)
    {
      bob::core::info << "# CGLogReg Training terminated: convergence after " << iter << " iterations." << std::endl;
      break;
    }
    // Terminates if maximum number of iterations has been reached
    if(m_max_iterations > 0 && iter+1 >= m_max_iterations)
    {
      bob::core::info << "# CGLogReg terminated: maximum number of iterations (" << m_max_iterations << ") reached." << std::endl;
      break;
    }

    // Backup previous values
    g_old = g;
    w_old = w;
  }

  // Updates the LinearMachine
  machine.resize(n_features, 1);
  machine.setInputSubtraction(mean);
  machine.setInputDivision(std_dev);

  blitz::Array<double,2>& w_ = machine.updateWeights();
  w_(rall,0) = w(rd); // Weights: first D values
  machine.setBiases(w(n_features)); // Bias: D+1 value
}

