/// @file MAP_GMMTrainer.h
/// @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
/// @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
/// @brief This class implements the maximum a posteriori M-step of the expectation-maximisation algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation.
/// @details See Section 3.4 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000. We use a "single adaptation coefficient", alpha_i, and thus a single relevance factor, r.

#ifndef _MAP_GMMTRAINER_H
#define _MAP_GMMTRAINER_H

#include "GMMTrainer.h"
#include <limits>
#include <core/Exception.h>

namespace Torch {
namespace trainer {

/// @brief This class implements the maximum a posteriori M-step of the expectation-maximisation algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation.
/// @details See Section 3.4 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000. We use a "single adaptation coefficient", alpha_i, and thus a single relevance factor, r.
class MAP_GMMTrainer : public GMMTrainer {
  public:

    /// Default constructor
    MAP_GMMTrainer(double relevance_factor = 0, bool update_means = true, bool update_variances = false, 
      bool update_weights = false, double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon());

    /// Destructor
    virtual ~MAP_GMMTrainer();

    /// Set the GMM to use as a prior for MAP adaptation.
    /// Generally, this is a "universal background model" (UBM),
    /// also referred to as a "world model".
    bool setPriorGMM(Torch::machine::GMMMachine *prior_gmm);

    /// Performs a maximum a posteriori (MAP) update of the GMM parameters
    /// using the accumulated statistics in m_ss and the 
    /// parameters of the prior model
    /// Implements EMTrainer::mStep()
    void mStep(Torch::machine::GMMMachine& gmm, const Torch::io::Arrayset& data);

    /// Use a Torch3-like adaptation rule rather than Reynolds'one
    /// In this case, alpha is a configuration variable rather than a function of the zeroth 
    /// order statistics and a relevance factor (should be in range [0,1])
    void setT3MAP(const double alpha) { m_T3_adaptation = true; m_T3_alpha = alpha; }
    void unsetT3MAP() { m_T3_adaptation = false; }
    
  protected:

    /// The relevance factor for MAP adaptation, r (see Reynolds et al., \"Speaker Verification Using Adapted Gaussian Mixture Models\", Digital Signal Processing, 2000).
    double relevance_factor;
    
    /// The GMM to use as a prior for MAP adaptation.
    /// Generally, this is a "universal background model" (UBM),
    /// also referred to as a "world model"
    Torch::machine::GMMMachine *m_prior_gmm;

    /// The alpha for the Torch3-like adaptation
    double m_T3_alpha;
    /// Whether Torch3-like adaptation should be used or not
    bool m_T3_adaptation;

  private:

    /// cache to avoid re-allocation
    mutable blitz::Array<double,1> m_cache_alpha;
    mutable blitz::Array<double,1> m_cache_ml_weights;
    mutable blitz::Array<double,1> m_cache_prior_weights;
    mutable blitz::Array<double,1> m_cache_new_weights;
    mutable blitz::Array<double,2> m_cache_ml_means;
    mutable blitz::Array<double,2> m_cache_prior_means;
    mutable blitz::Array<double,2> m_cache_new_means;
    mutable blitz::Array<double,2> m_cache_prior_variances;
    mutable blitz::Array<double,2> m_cache_Exx;
    mutable blitz::Array<double,2> m_cache_means;
    mutable blitz::Array<double,2> m_cache_new_variances;

};

}}

#endif
