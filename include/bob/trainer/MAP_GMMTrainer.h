/**
 * @file bob/trainer/MAP_GMMTrainer.h
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * @brief This class implements the maximum a posteriori M-step of the expectation-maximisation algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation.
 * @details See Section 3.4 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000. We use a "single adaptation coefficient", alpha_i, and thus a single relevance factor, r.
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

#ifndef BOB_TRAINER_MAP_GMMTRAINER_H
#define BOB_TRAINER_MAP_GMMTRAINER_H

#include <bob/trainer/GMMTrainer.h>
#include <limits>
#include <bob/core/Exception.h>

namespace bob { namespace trainer {
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief This class implements the maximum a posteriori M-step of the expectation-maximisation algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation.
 * @details See Section 3.4 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000. We use a "single adaptation coefficient", alpha_i, and thus a single relevance factor, r.
 */
class MAP_GMMTrainer: public GMMTrainer 
{
  public:
    /**
     * @brief Default constructor
     */
    MAP_GMMTrainer(const double relevance_factor=0, const bool update_means=true, 
      const bool update_variances=false, const bool update_weights=false, 
      const double mean_var_update_responsibilities_threshold = 
        std::numeric_limits<double>::epsilon());

    /**
     * @brief Copy constructor
     */
    MAP_GMMTrainer(const MAP_GMMTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~MAP_GMMTrainer();

    /**
     * @brief Initialization
     */
    virtual void initialize(bob::machine::GMMMachine& gmm,
      const blitz::Array<double,2>& data);

    /**
     * @brief Assigns from a different MAP_GMMTrainer
     */
    MAP_GMMTrainer& operator=(const MAP_GMMTrainer &other);

    /**
     * @brief Equal to
     */
    bool operator==(const MAP_GMMTrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const MAP_GMMTrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const MAP_GMMTrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;
 
    /**
     * @brief Set the GMM to use as a prior for MAP adaptation.
     * Generally, this is a "universal background model" (UBM),
     * also referred to as a "world model".
     */
    bool setPriorGMM(boost::shared_ptr<bob::machine::GMMMachine> prior_gmm);

    /**
     * @brief Performs a maximum a posteriori (MAP) update of the GMM
     * parameters using the accumulated statistics in m_ss and the 
     * parameters of the prior model
     * Implements EMTrainer::mStep()
     */
    void mStep(bob::machine::GMMMachine& gmm,
      const blitz::Array<double,2>& data);

    /**
     * @brief Use a Torch3-like adaptation rule rather than Reynolds'one
     * In this case, alpha is a configuration variable rather than a function of the zeroth 
     * order statistics and a relevance factor (should be in range [0,1])
     */
    void setT3MAP(const double alpha) { m_T3_adaptation = true; m_T3_alpha = alpha; }
    void unsetT3MAP() { m_T3_adaptation = false; }
    
  protected:

    /**
     * The relevance factor for MAP adaptation, r (see Reynolds et al., \"Speaker Verification Using Adapted Gaussian Mixture Models\", Digital Signal Processing, 2000).
     */
    double m_relevance_factor;
    
    /**
     * The GMM to use as a prior for MAP adaptation.
     * Generally, this is a "universal background model" (UBM),
     * also referred to as a "world model"
     */
    boost::shared_ptr<bob::machine::GMMMachine> m_prior_gmm;

    /**
     * The alpha for the Torch3-like adaptation
     */
    double m_T3_alpha;
    /**
     * Whether Torch3-like adaptation should be used or not
     */
    bool m_T3_adaptation;

  private:
    /// cache to avoid re-allocation
    mutable blitz::Array<double,1> m_cache_alpha;
    mutable blitz::Array<double,1> m_cache_ml_weights;
};

/**
 * @}
 */
}}

#endif
