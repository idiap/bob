/// @file ML_GMMTrainer.h
/// @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
/// @brief This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.
/// @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006

#ifndef _ML_GMMTRAINER_H
#define _ML_GMMTRAINER_H

#include "GMMTrainer.h"

namespace Torch {
namespace trainer {
/// @brief This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.
/// @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006
class ML_GMMTrainer : public GMMTrainer {
  public:

    /// Default constructor
    ML_GMMTrainer();

    /// Destructor
    virtual ~ML_GMMTrainer();

  protected:

    /// Performs a maximum likelihood (ML) update of the GMM parameters
    /// using the accumulated statistics in m_ss
    /// Implements EMTrainer::mStep()
    void mStep (Torch::machine::GMMMachine& gmm, const Sampler<Torch::machine::FrameSample>& data);
};

}
}

#endif
