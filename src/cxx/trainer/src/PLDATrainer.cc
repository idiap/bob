/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue 11 Oct 2011
 *
 * @brief Probabilistic Linear Discriminant Analysis
 */

#include <algorithm>
#include <boost/random.hpp>
#include <vector>

#include "trainer/PLDATrainer.h"
#include "core/array_copy.h"
#include "math/linear.h"
#include "math/lu_det.h"
#include "trainer/Exception.h"


namespace tca = Torch::core::array;
namespace io = Torch::io;
namespace mach = Torch::machine;
namespace math = Torch::math;
namespace train = Torch::trainer;

train::PLDABaseTrainer::PLDABaseTrainer(int nf, int ng, 
    double convergence_threshold, int max_iterations, bool compute_likelihood):
  EMTrainerNew<mach::PLDABaseMachine, std::vector<io::Arrayset> >
    (convergence_threshold, max_iterations, compute_likelihood), 
  m_nf(nf), m_ng(ng), m_S(0,0),
  m_z_first_order(0), m_sum_z_second_order(0,0),
  m_seed(-1),
  m_y_first_order(0), m_y_second_order(0),
  m_n_samples_per_id(0), m_n_samples_in_training(), m_B(0,0),
  m_Ft_isigma_G(0,0), m_eta(0,0), m_zeta(), m_iota(),
  m_cache_nf(0), m_cache_D_1(0), m_cache_D_2(0), m_cache_for_y_first_order(),
  m_cache_nfng_nfng(0,0), m_cache_D_nfng_1(0,0), m_cache_D_nfng_2(0,0)
{
}

train::PLDABaseTrainer::PLDABaseTrainer(const train::PLDABaseTrainer& other):
  EMTrainerNew<mach::PLDABaseMachine, std::vector<io::Arrayset> >
    (other.m_convergence_threshold, other.m_max_iterations, 
     other.m_compute_likelihood),
  m_nf(other.m_nf), m_ng(other.m_ng),
  m_S(tca::ccopy(other.m_S)),
  m_z_first_order(),
  m_sum_z_second_order(tca::ccopy(other.m_sum_z_second_order)),
  m_seed(other.m_seed),
  m_y_first_order(0), m_y_second_order(0),
  m_n_samples_per_id(other.m_n_samples_per_id),
  m_n_samples_in_training(other.m_n_samples_in_training), 
  m_B(tca::ccopy(other.m_B)), 
  m_Ft_isigma_G(tca::ccopy(other.m_Ft_isigma_G)), 
  m_eta(tca::ccopy(other.m_eta)), 
  m_cache_nf(tca::ccopy(other.m_cache_nf)),
  m_cache_D_1(tca::ccopy(other.m_cache_D_1)),
  m_cache_D_2(tca::ccopy(other.m_cache_D_2)),
  m_cache_for_y_first_order(),
  m_cache_nfng_nfng(tca::ccopy(other.m_cache_nfng_nfng)),
  m_cache_D_nfng_1(tca::ccopy(other.m_cache_D_nfng_1)),
  m_cache_D_nfng_2(tca::ccopy(other.m_cache_D_nfng_2))
{
  tca::ccopy(other.m_z_first_order, m_z_first_order);
  tca::ccopy(other.m_y_first_order, m_y_first_order);
  tca::ccopy(other.m_y_second_order, m_y_second_order);
  tca::ccopy(other.m_zeta, m_zeta);
  tca::ccopy(other.m_iota, m_iota);
  tca::ccopy(other.m_cache_for_y_first_order, m_cache_for_y_first_order);
}

train::PLDABaseTrainer::~PLDABaseTrainer() {}

train::PLDABaseTrainer& train::PLDABaseTrainer::operator=
(const train::PLDABaseTrainer& other) 
{
  m_convergence_threshold = other.m_convergence_threshold;
  m_max_iterations = other.m_max_iterations;
  m_compute_likelihood = other.m_compute_likelihood;
  m_nf = other.m_nf;
  m_ng = other.m_ng;
  m_S = tca::ccopy(other.m_S);
  tca::ccopy(other.m_z_first_order, m_z_first_order);
  m_sum_z_second_order = tca::ccopy(other.m_sum_z_second_order);
  m_seed = other.m_seed;
  tca::ccopy(other.m_y_first_order, m_y_first_order);
  tca::ccopy(other.m_y_second_order, m_y_second_order);
  m_n_samples_per_id = other.m_n_samples_per_id;
  m_n_samples_in_training = other.m_n_samples_in_training;
  m_B = tca::ccopy(other.m_B); 
  m_Ft_isigma_G = tca::ccopy(other.m_Ft_isigma_G); 
  m_eta = tca::ccopy(other.m_eta); 
  tca::ccopy(other.m_iota, m_iota);
  m_cache_nf = tca::ccopy(other.m_cache_nf);
  m_cache_D_1 = tca::ccopy(other.m_cache_D_1);
  m_cache_D_2 = tca::ccopy(other.m_cache_D_2);
  tca::ccopy(other.m_cache_for_y_first_order, m_cache_for_y_first_order);
  m_cache_nfng_nfng = tca::ccopy(other.m_cache_nfng_nfng);
  m_cache_D_nfng_1 = tca::ccopy(other.m_cache_D_nfng_1);
  m_cache_D_nfng_2 = tca::ccopy(other.m_cache_D_nfng_2);
  return *this;
}

void train::PLDABaseTrainer::initialization(mach::PLDABaseMachine& machine,
  const std::vector<io::Arrayset>& v_ar) 
{
  // Checks training data
  checkTrainingData(v_ar);

  // Gets dimension (first Arrayset)
  size_t n_features = v_ar[0].getShape()[0];
  // Resizes the PLDABaseMachine
  machine.resize(n_features, m_nf, m_ng);

  // Reinitializes array members
  initMembers(v_ar);

  // Computes the mean and the covariance if required
  computeMeanVariance(machine, v_ar);

  // TODO: add alternative initialization (e.g. using scatter)
  // Random initialization of F, G and Sigma
  initRandomFGSigma(machine);
}

void train::PLDABaseTrainer::finalization(mach::PLDABaseMachine& machine,
  const std::vector<io::Arrayset>& v_ar) 
{
  // Precomputes constant parts of the log likelihood and (gamma_a)
  precomputeLogLike(machine, v_ar);
  // Adds the case 1 sample if not already done (always used for scoring)
  machine.getAddGamma(1);
  machine.getAddLogLikeConstTerm(1);
}

void train::PLDABaseTrainer::checkTrainingData(const std::vector<io::Arrayset>& v_ar)
{
  // Checks that the vector of Arraysets is not empty
  if(v_ar.size() == 0)
    throw Torch::trainer::EmptyTrainingSet();

  // Gets dimension (first Arrayset)
  size_t n_features = v_ar[0].getShape()[0];
  // Checks dimension consistency
  for(size_t i=0; i<v_ar.size(); ++i) {
    // Checks for arrayset data type and shape
    if(v_ar[i].getElementType() != Torch::core::array::t_float64) {
      throw Torch::io::TypeError(v_ar[i].getElementType(),
        Torch::core::array::t_float64);
    }
    if(v_ar[i].getNDim() != 1) {
      throw Torch::io::DimensionError(v_ar[i].getNDim(), 1);
    }
    if(v_ar[i].getShape()[0] != n_features)
      throw Torch::trainer::WrongNumberOfFeatures(v_ar[i].getShape()[0], 
                                                  n_features, i);
  } 
}

void train::PLDABaseTrainer::initMembers(const std::vector<io::Arrayset>& v_ar)
{
  // Gets dimension (first Arrayset)
  size_t n_features = v_ar[0].getShape()[0]; // dimensionality of the data
  size_t n_identities = v_ar.size();

  m_S.resize(n_features,n_features);
  m_sum_z_second_order.resize(m_nf+m_ng, m_nf+m_ng);

  // Loops over the identities
  for(size_t i=0; i<n_identities; ++i) 
  {
    // Number of training samples for this identity
    size_t n_i = v_ar[i].size(); 
    // m_z_first_order
    blitz::Array<double,2> z_i(n_i, m_nf+m_ng);
    m_z_first_order.push_back(z_i);

    size_t q_i = m_nf + n_i * m_ng;
    // m_y_{first,second}_order
    blitz::Array<double,1> yi1(q_i);
    m_y_first_order.push_back(yi1);
    blitz::Array<double,2> yi2(q_i, q_i);
    m_y_second_order.push_back(yi2);
    // m_cache_for_y_{first,second}_order
    m_cache_for_y_first_order[n_i].reference(blitz::Array<double,1>(q_i));

    // m_n_samples_per_id
    m_n_samples_per_id.push_back(n_i);
    
    // Maps dependent on the number of samples per identity
    std::map<size_t,bool>::iterator it;
    it = m_n_samples_in_training.find(n_i);
    if(it == m_n_samples_in_training.end())
    {
      // Indicates if there are identities with n_i training samples and if
      // corresponding matrices are up to date.
      m_n_samples_in_training[n_i] = false;
      // Allocates arrays for identities with n_i training samples
      m_zeta[n_i].reference(blitz::Array<double,2>(m_ng, m_ng));
      m_iota[n_i].reference(blitz::Array<double,2>(m_nf, m_ng));
    }
  }

  m_B.resize(n_features, m_nf+m_ng);
  m_Ft_isigma_G.resize(m_nf,m_ng);
  m_eta.resize(m_nf,m_ng);

  // Cache
  m_cache_nf.resize(m_nf);
  m_cache_D_1.resize(n_features);
  m_cache_D_2.resize(n_features);
  m_cache_nfng_nfng.resize(m_nf+m_ng,m_nf+m_ng);
  m_cache_D_nfng_1.resize(n_features,m_nf+m_ng);
  m_cache_D_nfng_2.resize(n_features,m_nf+m_ng);
}

void train::PLDABaseTrainer::computeMeanVariance(mach::PLDABaseMachine& machine, 
  const std::vector<io::Arrayset>& v_ar) 
{
  blitz::Array<double,1>& mu = machine.updateMu();
  // TODO: Uncomment variance computation if required
  /*  if(m_compute_likelihood) 
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
  else */
  {
    // Computes the mean and updates mu
    mu = 0.;
    size_t n_samples = 0;
    for(size_t j=0; j<v_ar.size(); ++j) {
      n_samples += v_ar[j].size();
      for (size_t i=0; i<v_ar[j].size(); ++i)
        mu += v_ar[j].get<double,1>(i);
    }
    mu /= static_cast<double>(n_samples);
  }
}

void train::PLDABaseTrainer::initRandomFGSigma(mach::PLDABaseMachine& machine)
{
  // Initializes the random number generator
  boost::mt19937 rng;
  if(m_seed != -1)
    rng.seed((uint32_t)m_seed);
  boost::normal_distribution<> range_n;
  boost::uniform_01<> range_01;
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > 
    die_n(rng, range_n);
  boost::variate_generator<boost::mt19937&, boost::uniform_01<> > 
    die_01(rng, range_01);
    
  double ratio = 1.; // TODO: check if a ratio is required
  // F initialization
  blitz::Array<double,2>& F = machine.updateF();
  for(int j=0; j<F.extent(0); ++j)
    for(int i=0; i<F.extent(1); ++i)
      F(j,i) = die_n() * ratio;
  // G initialization
  blitz::Array<double,2>& G = machine.updateG();
  for(int j=0; j<G.extent(0); ++j)
    for(int i=0; i<G.extent(1); ++i)
      G(j,i) = die_n() * ratio;
  // sigma2 initialization
  blitz::Array<double,1>& sigma = machine.updateSigma();
  double eps = 1e-5; // Sigma should be invertible...
  for(int j=0; j<sigma.extent(0); ++j)
    sigma(j) = die_01() * ratio + eps;

  // Precompute values
  machine.precompute();
}


void train::PLDABaseTrainer::eStep(mach::PLDABaseMachine& machine, 
  const std::vector<io::Arrayset>& v_ar)
{  
  // Precomputes useful variables using current estimates of F,G, and sigma
  precomputeFromFGSigma(machine);
  // Gets the mean mu from the machine
  const blitz::Array<double,1>& mu = machine.getMu();
  const blitz::Array<double,2>& alpha = machine.getAlpha();
  const blitz::Array<double,2>& F = machine.getF();
  const blitz::Array<double,2>& FtBeta = machine.getFtBeta();
  const blitz::Array<double,2>& GtISigma = machine.getGtISigma();

  // Initializes sum of z second order statistics to 0
  m_sum_z_second_order = 0.;
  for(size_t i=0; i<v_ar.size(); ++i)
  {
    // 1/ First order statistics of y
    blitz::Array<double,1>& y_first_order_i = m_y_first_order[i];
    blitz::Array<double,1>& cache_for_y =  m_cache_for_y_first_order[v_ar[i].size()];
    // Loop over the samples
    cache_for_y = 0.;
    // Computes expectation of y_i = [h_i w_i1 ... w_iJ]
    // 1/a/ Computes expectation of h_i
    for(size_t j=0; j<v_ar[i].size(); ++j)
    {
      // m_cache_D_1 = x_sj-mu
      m_cache_D_1 = v_ar[i].get<double,1>(j) - mu;

      // m_cache_nf = F^T.beta.(x_sj-mu)
      Torch::math::prod(FtBeta, m_cache_D_1, m_cache_nf);
      // cache_for_y(0,nf-1) = sum_j F^T.beta.(x_sj-mu)
      cache_for_y(blitz::Range(0, m_nf-1)) += m_cache_nf;
    }
    const blitz::Array<double,2>& gamma_a = machine.getAddGamma(v_ar[i].size());
    blitz::Range r_hi(0, m_nf-1);
    blitz::Array<double,1> cache_for_y_slice = cache_for_y(r_hi);
    blitz::Array<double,1> y_first_order_i_slice = y_first_order_i(r_hi);
    // y_first_order_i_slice = gamma_A  sum_j F^T.beta.(x_sj-mu)
    Torch::math::prod(gamma_a, cache_for_y_slice, y_first_order_i_slice);

    // 1/b/ Computes expectation of w_ij
    // m_cache_D_2 = F.E{h_i}
    Torch::math::prod(F, y_first_order_i_slice, m_cache_D_2);
    for(size_t j=0; j<v_ar[i].size(); ++j)
    {
      // m_cache_D_1 = x_sj - mu - F.E{h_i}
      m_cache_D_1 = v_ar[i].get<double,1>(j) - mu - m_cache_D_2;
      // y_i_slice = G^T.sigma^-1.(x_sj-mu-fhi)
      blitz::Range r_sample_j(m_nf + j*m_ng, m_nf + (j+1)*m_ng -1);
      blitz::Array<double,1> cache_for_y_slice = cache_for_y(r_sample_j);
      Torch::math::prod(GtISigma, m_cache_D_1, cache_for_y_slice);
      // y_first_order_i_slice = (Id+G^T.sigma^-1.G)^-1.G^T.sigma^-1.(x_sj-mu)
      blitz::Array<double,1> y_first_order_i_slice = y_first_order_i(r_sample_j);
      Torch::math::prod(alpha, cache_for_y_slice, y_first_order_i_slice);
    }
    
    // TODO: improve second order statistics computation 
    //       (some parts if y_second_order_i are currently useless)
    // 2/ Second order statistics of y
    blitz::Array<double,2>& y_second_order_i = m_y_second_order[i];
    // y_second_order_i = E{y_i}.E{y_i}^T
    Torch::math::prod(y_first_order_i, y_first_order_i, y_second_order_i);
   
    // Precomputed values 
    blitz::Array<double,2>& zeta_a = m_zeta[v_ar[i].size()];
    blitz::Array<double,2>& iota_a = m_iota[v_ar[i].size()];
    blitz::Array<double,2> iotat_a = iota_a.transpose(1,0);

    // Extracts statistics of z_ij = [h_i w_ij] from y_i = [h_i w_i1 ... w_iJ]
    blitz::Range r1(0, m_nf-1);
    blitz::Range r2(m_nf, m_nf+m_ng-1);
    for(size_t j=0; j<v_ar[i].size(); ++j)
    {
      // 1/ First order statistics of z
      blitz::Array<double,1> z_first_order_ij_1 = m_z_first_order[i](j,r1);
      z_first_order_ij_1 = y_first_order_i(r1); // h_i
      blitz::Array<double,1> z_first_order_ij_2 = m_z_first_order[i](j,r2);
      blitz::Range rj(m_nf + j*m_ng, m_nf + (j+1)*m_ng-1);
      z_first_order_ij_2 = y_first_order_i(rj); // w_ij

      // 2/ Second order statistics of z
      blitz::Array<double,2> z_so_11 = m_sum_z_second_order(r1,r1);
      z_so_11 += gamma_a + y_second_order_i(r1,r1); 
      blitz::Array<double,2> z_so_12 = m_sum_z_second_order(r1,r2);
      z_so_12 += iota_a + y_second_order_i(r1,rj);
      blitz::Array<double,2> z_so_21 = m_sum_z_second_order(r2,r1);
      z_so_21 += iotat_a + y_second_order_i(rj,r1);
      blitz::Array<double,2> z_so_22 = m_sum_z_second_order(r2,r2);
      z_so_22 += zeta_a + y_second_order_i(rj,rj);
    }
  }
}

void train::PLDABaseTrainer::precomputeFromFGSigma(mach::PLDABaseMachine& machine)
{
  // non const because of transpose() (compability with old blitz versions)  
  blitz::Array<double,2>& F = machine.updateF();
  blitz::Array<double,2> Ft = F.transpose(1,0);
  blitz::Array<double,2>& Gt_isigma = machine.updateGtISigma();
  blitz::Array<double,2> Gt_isigma_t = Gt_isigma.transpose(1,0);
  const blitz::Array<double,2>& alpha = machine.getAlpha();

  // blitz indices
  blitz::firstIndex i;
  blitz::secondIndex j;

  // Precomputes F, G and sigma-based expressions
  Torch::math::prod(Ft, Gt_isigma_t, m_Ft_isigma_G);
  Torch::math::prod(m_Ft_isigma_G, alpha, m_eta); 
  blitz::Array<double,2> etat = m_eta.transpose(1,0);

  // Reinitializes all the zeta_a and iota_a
  std::map<size_t,bool>::iterator it;
  for(it=m_n_samples_in_training.begin(); it!=m_n_samples_in_training.end(); 
      ++it)
    it->second = false;

  for(it=m_n_samples_in_training.begin(); it!=m_n_samples_in_training.end(); 
      ++it)
  {
    size_t n_i = it->first;
    // Precomputes zeta and iota for identities with q_i training samples,
    // if not already done
    if(!it->second)
    {
      blitz::Array<double,2>& gamma_a = machine.getAddGamma(n_i);
      blitz::Array<double,2>& zeta_a = m_zeta[n_i];
      blitz::Array<double,2>& iota_a = m_iota[n_i];
      Torch::math::prod(gamma_a, m_eta, iota_a);
      Torch::math::prod(etat, iota_a, zeta_a);
      zeta_a += alpha;
      iota_a = - iota_a;
      // Now up to date
      it->second = true;
    }
  }
}

void train::PLDABaseTrainer::precomputeLogLike(mach::PLDABaseMachine& machine, 
  const std::vector<io::Arrayset>& v_ar) 
{
  // Precomputes the log determinant of alpha and sigma
  machine.precomputeLogLike();

  // Precomputes the log likelihood constant term
  std::map<size_t,bool>::iterator it;
  for(it=m_n_samples_in_training.begin(); it!=m_n_samples_in_training.end(); 
      ++it)
  {
    // Precomputes the log likelihood constant term for identities with q_i 
    // training samples, if not already done
    machine.getAddLogLikeConstTerm(it->first);
  }
}


void train::PLDABaseTrainer::mStep(mach::PLDABaseMachine& machine, 
  const std::vector<io::Arrayset>& v_ar) 
{
  // TODO: 0/ Add mean update rule as an option?

  // 1/ New estimate of B = {F G}
  updateFG(machine, v_ar);

  // 2/ New estimate of Sigma
  updateSigma(machine, v_ar);

  // 3/ Precomputes new values after updating F, G and sigma
  machine.precompute();
  // Precomputes useful variables using current estimates of F,G, and sigma
  precomputeFromFGSigma(machine);
}

void train::PLDABaseTrainer::updateFG(mach::PLDABaseMachine& machine,
  const std::vector<io::Arrayset>& v_ar)
{
  /// Computes the B matrix (B = [F G])
  /// B = (sum_ij (x_ij-mu).E{z_i}^T).(sum_ij E{z_i.z_i^T})^-1

  // 1/ Computes the numerator (sum_ij (x_ij-mu).E{z_i}^T)
  // Gets the mean mu from the machine
  const blitz::Array<double,1>& mu = machine.getMu();
  m_cache_D_nfng_2 = 0.;
  for(size_t i=0; i<v_ar.size(); ++i)
  {
    // Loop over the samples
    for(size_t j=0; j<v_ar[i].size(); ++j)
    {
      // m_cache_D_1 = x_sj-mu
      m_cache_D_1 = v_ar[i].get<double,1>(j) - mu;
      // z_first_order_ij = E{z_ij}
      blitz::Array<double,1> z_first_order_ij = m_z_first_order[i](j, blitz::Range::all());
      // m_cache_D_nfng_1 = (x_sj-mu).E{z_ij}^T
      Torch::math::prod(m_cache_D_1, z_first_order_ij, m_cache_D_nfng_1);
      m_cache_D_nfng_2 += m_cache_D_nfng_1;
    }
  }

  // 2/ Computes the denominator inv(sum_ij E{z_i.z_i^T})
  Torch::math::inv(m_sum_z_second_order, m_cache_nfng_nfng);

  // 3/ Computes numerator / denominator
  Torch::math::prod(m_cache_D_nfng_2, m_cache_nfng_nfng, m_B);

  // 4/ Updates the machine 
  // TODO: Use B as cache in the trainer, and only sets F and G when calling
  //       finalization()
  blitz::Array<double, 2>& F = machine.updateF();
  blitz::Array<double, 2>& G = machine.updateG();
  F = m_B(blitz::Range::all(), blitz::Range(0,m_nf-1));
  G = m_B(blitz::Range::all(), blitz::Range(m_nf,m_nf+m_ng-1));
}

void train::PLDABaseTrainer::updateSigma(mach::PLDABaseMachine& machine,
  const std::vector<io::Arrayset>& v_ar)
{
  /// Computes the Sigma matrix
  /// Sigma = 1/IJ sum_ij Diag{(x_ij-mu).(x_ij-mu)^T - B.E{z_i}.(x_ij-mu)^T}

  // Gets the mean mu and the matrix sigma from the machine
  blitz::Array<double,1>& sigma = machine.updateSigma();
  const blitz::Array<double,1>& mu = machine.getMu();

  sigma = 0.;
  size_t n_IJ=0; /// counts the number of samples
  for(size_t i=0; i<v_ar.size(); ++i)
  {
    // Loop over the samples
    for(size_t j=0; j<v_ar[i].size(); ++j)
    {
      // m_cache_D_1 = x_ij-mu
      m_cache_D_1 = v_ar[i].get<double,1>(j) - mu;
      // sigma += Diag{(x_ij-mu).(x_ij-mu)^T}
      sigma += blitz::pow2(m_cache_D_1);

      // z_first_order_ij = E{z_ij}
      blitz::Array<double,1> z_first_order_ij = m_z_first_order[i](j, blitz::Range::all());
      // m_cache_D_2 = B.E{z_ij}
      Torch::math::prod(m_B, z_first_order_ij, m_cache_D_2);
      // sigma -= Diag{B.E{z_ij}.(x_ij-mu)
      sigma -= (m_cache_D_1 * m_cache_D_2);
      ++n_IJ;
    }
  }
  // Normalizes by the number of samples
  sigma /= static_cast<double>(n_IJ);
}

double train::PLDABaseTrainer::computeLikelihood(mach::PLDABaseMachine& machine,
  const std::vector<io::Arrayset>& v_ar)
{
  double llh = 0.;
  // TODO: implement log likelihood computation
  return llh;
}


train::PLDATrainer::PLDATrainer(mach::PLDAMachine& plda_machine, 
    train::PLDABaseTrainer& base_trainer): 
  m_plda_machine(plda_machine),
  m_base_trainer(base_trainer),
  m_cache_D_1(plda_machine.getDimD()),
  m_cache_D_2(plda_machine.getDimD()),
  m_cache_nf_1(plda_machine.getDimF())
{
}

train::PLDATrainer::PLDATrainer(const train::PLDATrainer& other):
  m_plda_machine(other.m_plda_machine),
  m_base_trainer(other.m_base_trainer),
  m_cache_D_1(tca::ccopy(other.m_cache_D_1)),
  m_cache_D_2(tca::ccopy(other.m_cache_D_2)),
  m_cache_nf_1(tca::ccopy(other.m_cache_nf_1))
{
}

train::PLDATrainer::~PLDATrainer() {
}

train::PLDATrainer& train::PLDATrainer::operator=
(const train::PLDATrainer& other) 
{
  m_plda_machine = other.m_plda_machine;
  m_base_trainer = other.m_base_trainer;
  m_cache_D_1.reference(tca::ccopy(other.m_cache_D_1));
  m_cache_D_2.reference(tca::ccopy(other.m_cache_D_2));
  m_cache_nf_1.reference(tca::ccopy(other.m_cache_nf_1));
  return *this;
}


void train::PLDATrainer::enrol(const io::Arrayset& ar)
{
  // Checks Arrayset using the check function from the PLDABaseTrainer
  // Gets dimension (first Arrayset)
  size_t n_features = ar.getShape()[0];
  size_t n_samples = ar.size();
    
  // Checks for arrayset data type and shape
  if(ar.getElementType() != Torch::core::array::t_float64) {
    throw Torch::io::TypeError(ar.getElementType(),
      Torch::core::array::t_float64);
  }
  if(ar.getNDim() != 1) {
    throw Torch::io::DimensionError(ar.getNDim(), 1);
  }
  // TODO: Do a useful comparison against the dimensionality from the base 
  // trainer/machine
  if(ar.getShape()[0] != n_features)
    throw Torch::trainer::WrongNumberOfFeatures(ar.getShape()[0], 
                                                n_features, 0);

  // Useful values from the base machine
  blitz::Array<double, 1>& weighted_sum = m_plda_machine.updateWeightedSum();
  const blitz::Array<double, 1>& mu = m_plda_machine.getPLDABase()->getMu();
  const blitz::Array<double, 2>& beta = m_plda_machine.getPLDABase()->getBeta();
  const blitz::Array<double, 2>& FtBeta = m_plda_machine.getPLDABase()->getFtBeta();

  // Resizes the PLDA machine
  m_plda_machine.resize(m_plda_machine.getDimD(), m_plda_machine.getDimF(), 
    m_plda_machine.getDimG());

  // Updates the PLDA machine
  m_plda_machine.setNSamples(n_samples);
  double terma = 0.;
  weighted_sum = 0.;
  for(size_t i=0; i<n_samples; ++i) {
    m_cache_D_1 =  ar.get<double,1>(i) - mu;
    // a/ weighted sum
    Torch::math::prod(FtBeta, m_cache_D_1, m_cache_nf_1);
    weighted_sum += m_cache_nf_1;
    // b/ first xi dependent term of the log likelihood
    Torch::math::prod(beta, m_cache_D_1, m_cache_D_2);
    terma += -1 / 2. * blitz::sum(m_cache_D_1 * m_cache_D_2);
  }
  m_plda_machine.setWSumXitBetaXi(terma);

  // Adds the precomputed values for the cases N and N+1 if not already 
  // in the base machine (used by the forward function, 1 already added)
  m_plda_machine.getAddGamma(n_samples);
  m_plda_machine.getAddLogLikeConstTerm(n_samples);
  m_plda_machine.getAddGamma(n_samples+1);
  m_plda_machine.getAddLogLikeConstTerm(n_samples+1);
  m_plda_machine.setLogLikelihood(m_plda_machine.computeLikelihood(
                                    blitz::Array<double,2>(0,0),true));
}
