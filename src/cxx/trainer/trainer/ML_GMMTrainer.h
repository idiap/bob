/// @file ML_GMMTrainer.h
/// @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
/// @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
/// @brief This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.
/// @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006

#ifndef _ML_GMMTRAINER_H
#define _ML_GMMTRAINER_H

#include "GMMTrainer.h"
#include <limits>

namespace Torch {
namespace trainer {
/// @brief This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.
/// @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006
class ML_GMMTrainer : public GMMTrainer {
  public:

    /// Default constructor
    ML_GMMTrainer(bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon());

    /// Destructor
    virtual ~ML_GMMTrainer();

    /// Performs a maximum likelihood (ML) update of the GMM parameters
    /// using the accumulated statistics in m_ss
    /// Implements EMTrainer::mStep()
    void mStep (Torch::machine::GMMMachine& gmm, const Torch::io::Arrayset& data);
};

}
}

#endif
