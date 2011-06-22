/// @file MAP_GMMTrainer.h
/// @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
/// @brief This class implements the maximum a posteriori M-step of the expectation-maximisation algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation.
/// @details See Section 3.4 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000. We use a "single adaptation coefficient", alpha_i, and thus a single relevance factor, r.

#ifndef _MAP_GMMTRAINER_H
#define _MAP_GMMTRAINER_H

#include "GMMTrainer.h"
#include <core/Exception.h>

namespace Torch {
namespace trainer {

/// @brief This class implements the maximum a posteriori M-step of the expectation-maximisation algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation.
/// @details See Section 3.4 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000. We use a "single adaptation coefficient", alpha_i, and thus a single relevance factor, r.
class MAP_GMMTrainer : public GMMTrainer {
  public:

    /// Default constructor
    MAP_GMMTrainer(double relevance_factor = 0, bool update_means = true, bool update_variances = false, bool update_weights = false);

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
    
  protected:

    /// The relevance factor for MAP adaptation, r (see Reynolds et al., \"Speaker Verification Using Adapted Gaussian Mixture Models\", Digital Signal Processing, 2000).
    double relevance_factor;
    
    /// The GMM to use as a prior for MAP adaptation.
    /// Generally, this is a "universal background model" (UBM),
    /// also referred to as a "world model"
    Torch::machine::GMMMachine *m_prior_gmm;
};

}
}

#endif
