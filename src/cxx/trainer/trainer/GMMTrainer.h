/// @file GMMTrainer.h
/// @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
/// @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
/// @brief This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.
/// @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006

#ifndef _GMMTRAINER_H
#define _GMMTRAINER_H

#include "io/Arrayset.h"
#include "trainer/EMTrainer.h"
#include "machine/GMMMachine.h"
#include "machine/GMMStats.h"
#include <limits>

namespace Torch {
namespace trainer {

/// @brief This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.
/// @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006
class GMMTrainer : public EMTrainer<Torch::machine::GMMMachine, Torch::io::Arrayset> {
  public:

    /// Default constructor
    GMMTrainer(bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon());
    
    /// Destructor
    virtual ~GMMTrainer();

    virtual void initialization(Torch::machine::GMMMachine& gmm, const Torch::io::Arrayset& data);
    
    /// Calculates and saves statistics across the dataset, 
    /// and saves these as m_ss. Calculates the average
    /// log likelihood of the observations given the GMM,
    /// and returns this in average_log_likelihood.
    /// 
    /// The statistics, m_ss, will be used in the mStep() that follows.
    /// Implements EMTrainer::eStep(double &)
    virtual double eStep(Torch::machine::GMMMachine& gmm, const Torch::io::Arrayset& data);

    
  protected:

    /// These are the sufficient statistics, calculated during the
    /// E-step and used during the M-step
    Torch::machine::GMMStats m_ss;
    
    /// update means on each iteration
    bool update_means;
    
    /// update variances on each iteration
    bool update_variances;
    
    /// update weights on each iteration
    bool update_weights;

    /// threshold over the responsibilities of the Gaussians
    /// Equations 9.24, 9.25 of Bishop, "Pattern recognition and machine learning", 2006
    /// require a division by the responsibilities, which might be equal to zero
    /// because of numerical issue. This threshold is used to avoid such divisions.
    double m_mean_var_update_responsibilities_threshold;
};

}
}
#endif
