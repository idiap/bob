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

namespace bob {
namespace trainer {
/**
 * @brief This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.
 * @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006
 */
class ML_GMMTrainer : public GMMTrainer {
  public:
    /**
     * Default constructor
     */
    ML_GMMTrainer(bool update_means = true, bool update_variances = false, bool update_weights = false,
      double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon());

    /**
     * Destructor
     */
    virtual ~ML_GMMTrainer();

    /**
     * Initialisation before the EM steps
     */
    virtual void initialization(bob::machine::GMMMachine& gmm, const blitz::Array<double,2>& data);

    /**
     * Performs a maximum likelihood (ML) update of the GMM parameters
     * using the accumulated statistics in m_ss
     * Implements EMTrainer::mStep()
     */
    void mStep(bob::machine::GMMMachine& gmm, const blitz::Array<double,2>& data);
  
  private:
    /**
     * Add cache to avoid re-allocation at each iteration
     */
    mutable blitz::Array<double,1> m_cache_ss_n_thresholded;
};

}}

#endif
