/**
 * @file bob/trainer/GMMTrainer.h
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * @brief This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.
 * @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006
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

#ifndef BOB_TRAINER_GMMTRAINER_H
#define BOB_TRAINER_GMMTRAINER_H

#include "EMTrainer.h"
#include <bob/machine/GMMMachine.h>
#include <bob/machine/GMMStats.h>
#include <limits>

namespace bob { namespace trainer {
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief This class implements the E-step of the expectation-maximisation 
 * algorithm for a GMM Machine.
 * @details See Section 9.2.2 of Bishop, 
 *   "Pattern recognition and machine learning", 2006
 */
class GMMTrainer: public EMTrainer<bob::machine::GMMMachine, blitz::Array<double,2> >
{
  public:
    /**
     * @brief Default constructor
     */
    GMMTrainer(const bool update_means=true, 
      const bool update_variances=false, const bool update_weights=false,
      const double mean_var_update_responsibilities_threshold =
        std::numeric_limits<double>::epsilon());
    
    /**
     * @brief Copy constructor
     */
    GMMTrainer(const GMMTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~GMMTrainer();

    /**
     * @brief Initialization before the EM steps
     */
    virtual void initialize(bob::machine::GMMMachine& gmm,
      const blitz::Array<double,2>& data);
    
    /**
     * @brief Calculates and saves statistics across the dataset,
     * and saves these as m_ss. Calculates the average
     * log likelihood of the observations given the GMM,
     * and returns this in average_log_likelihood.
     * 
     * The statistics, m_ss, will be used in the mStep() that follows.
     * Implements EMTrainer::eStep(double &)
     */
    virtual void eStep(bob::machine::GMMMachine& gmm,
      const blitz::Array<double,2>& data);

    /**
     * @brief Computes the likelihood using current estimates of the latent
     * variables
     */
    virtual double computeLikelihood(bob::machine::GMMMachine& gmm);

    /**
     * @brief Finalization after the EM steps
     */
    virtual void finalize(bob::machine::GMMMachine& gmm,
      const blitz::Array<double,2>& data);
  
    /**
     * @brief Assigns from a different GMMTrainer
     */
    GMMTrainer& operator=(const GMMTrainer &other);

    /**
     * @brief Equal to
     */
    bool operator==(const GMMTrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const GMMTrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const GMMTrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;
 
    /**
     * @brief Returns the internal GMM statistics. Useful to parallelize the 
     * E-step
     */
    const bob::machine::GMMStats& getGMMStats() const
    { return m_ss; }

    /**
     * @brief Sets the internal GMM statistics. Useful to parallelize the
     * E-step
     */
    void setGMMStats(const bob::machine::GMMStats& stats); 
     
  protected:
    /**
     * These are the sufficient statistics, calculated during the
     * E-step and used during the M-step
     */
    bob::machine::GMMStats m_ss;
    
    /**
     * update means on each iteration
     */
    bool m_update_means;
    
    /**
     * update variances on each iteration
     */
    bool m_update_variances;
    
    /**
     * update weights on each iteration
     */
    bool m_update_weights;

    /**
     * threshold over the responsibilities of the Gaussians
     * Equations 9.24, 9.25 of Bishop, "Pattern recognition and machine learning", 2006
     * require a division by the responsibilities, which might be equal to zero
     * because of numerical issue. This threshold is used to avoid such divisions.
     */
    double m_mean_var_update_responsibilities_threshold;
};

/**
 * @}
 */
}}

#endif
