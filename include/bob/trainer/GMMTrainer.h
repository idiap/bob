/**
 * @file bob/trainer/GMMTrainer.h
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * @brief This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.
 * @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006
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

#ifndef BOB_TRAINER_GMMTRAINER_H
#define BOB_TRAINER_GMMTRAINER_H

#include "bob/io/Arrayset.h"
#include "bob/trainer/EMTrainer.h"
#include "bob/machine/GMMMachine.h"
#include "bob/machine/GMMStats.h"
#include <limits>

namespace bob {
namespace trainer {

/**
 * @brief This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.
 * @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006
 */
class GMMTrainer : public EMTrainer<bob::machine::GMMMachine, bob::io::Arrayset> {
  public:

    /**
     * Default constructor
     */
    GMMTrainer(bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon());
    
    /**
     * Destructor
     */
    virtual ~GMMTrainer();

    /**
     * Initialization before the EM steps
     */
    virtual void initialization(bob::machine::GMMMachine& gmm, const bob::io::Arrayset& data);
    
    /**
     * Calculates and saves statistics across the dataset, 
     * and saves these as m_ss. Calculates the average
     * log likelihood of the observations given the GMM,
     * and returns this in average_log_likelihood.
     * 
     * The statistics, m_ss, will be used in the mStep() that follows.
     * Implements EMTrainer::eStep(double &)
     */
    virtual void eStep(bob::machine::GMMMachine& gmm, const bob::io::Arrayset& data);

    /**
     * Computes the likelihood using current estimates of the latent variables
     */
    virtual double computeLikelihood(bob::machine::GMMMachine& gmm);

    /**
     * Finalization after the EM steps
     */
    virtual void finalization(bob::machine::GMMMachine& gmm, const bob::io::Arrayset& data);
    
  protected:

    /**
     * These are the sufficient statistics, calculated during the
     * E-step and used during the M-step
     */
    bob::machine::GMMStats m_ss;
    
    /**
     * update means on each iteration
     */
    bool update_means;
    
    /**
     * update variances on each iteration
     */
    bool update_variances;
    
    /**
     * update weights on each iteration
     */
    bool update_weights;

    /**
     * threshold over the responsibilities of the Gaussians
     * Equations 9.24, 9.25 of Bishop, "Pattern recognition and machine learning", 2006
     * require a division by the responsibilities, which might be equal to zero
     * because of numerical issue. This threshold is used to avoid such divisions.
     */
    double m_mean_var_update_responsibilities_threshold;
};

}
}
#endif
