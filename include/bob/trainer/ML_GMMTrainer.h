/**
 * @file bob/trainer/ML_GMMTrainer.h
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * @brief This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.
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

#ifndef BOB_TRAINER_ML_GMMTRAINER_H
#define BOB_TRAINER_ML_GMMTRAINER_H

#include "GMMTrainer.h"
#include <limits>

namespace bob { namespace trainer {
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief This class implements the maximum likelihood M-step of the 
 *   expectation-maximisation algorithm for a GMM Machine.
 * @details See Section 9.2.2 of Bishop, 
 *  "Pattern recognition and machine learning", 2006
 */
class ML_GMMTrainer: public GMMTrainer {
  public:
    /**
     * @brief Default constructor
     */
    ML_GMMTrainer(const bool update_means=true, 
      const bool update_variances=false, const bool update_weights=false,
      const double mean_var_update_responsibilities_threshold = 
        std::numeric_limits<double>::epsilon());

    /**
     * @brief Copy constructor
     */
    ML_GMMTrainer(const ML_GMMTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~ML_GMMTrainer();

    /**
     * @brief Initialisation before the EM steps
     */
    virtual void initialize(bob::machine::GMMMachine& gmm,
      const blitz::Array<double,2>& data);

    /**
     * @brief Performs a maximum likelihood (ML) update of the GMM parameters
     * using the accumulated statistics in m_ss
     * Implements EMTrainer::mStep()
     */
    virtual void mStep(bob::machine::GMMMachine& gmm,
      const blitz::Array<double,2>& data);
 
    /**
     * @brief Assigns from a different ML_GMMTrainer
     */
    ML_GMMTrainer& operator=(const ML_GMMTrainer &other);

    /**
     * @brief Equal to
     */
    bool operator==(const ML_GMMTrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const ML_GMMTrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const ML_GMMTrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

 
  private:
    /**
     * @brief Add cache to avoid re-allocation at each iteration
     */
    mutable blitz::Array<double,1> m_cache_ss_n_thresholded;
};

/**
 * @}
 */
}}

#endif
