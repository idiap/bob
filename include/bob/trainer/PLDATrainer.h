/**
 * @file bob/trainer/PLDATrainer.h
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Probabilistic PLDA Discriminant Analysis implemented using
 * Expectation Maximization.
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

#ifndef BOB_TRAINER_PLDA_TRAINER_H
#define BOB_TRAINER_PLDA_TRAINER_H

#include "EMTrainer.h"
#include <bob/machine/PLDAMachine.h>
#include <blitz/array.h>
#include <map>
#include <vector>

namespace bob { namespace trainer { 
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief This class can be used to train the \f$F\f$, \f$G\f$ and 
 * \f$\Sigma\f$ matrices and the mean vector \f$\mu\f$ of a PLDA model.\n
 * References:\n
 * 1. 'A Scalable Formulation of Probabilistic Linear Discriminant Analysis: 
 *     Applied to Face Recognition', Laurent El Shafey, Chris McCool, 
 *     Roy Wallace, Sebastien Marcel, TPAMI'2013
 * 2. 'Probabilistic Linear Discriminant Analysis for Inference About 
 *     Identity', Prince and Elder, ICCV'2007\n
 * 3. 'Probabilistic Models for Inference about Identity', Li, Fu, Mohammed, 
 *     Elder and Prince, TPAMI'2012
 */
class PLDATrainer: public EMTrainer<bob::machine::PLDABase, 
                                        std::vector<blitz::Array<double,2> > >
{
  public: //api
    /**
     * @brief Default constructor.\n Initializes a new PLDA trainer. The 
     * training stage will place the resulting components in the 
     * PLDABase.
     */
    PLDATrainer(const size_t max_iterations=100, const bool use_sum_second_order=true);

    /**
     * @brief Copy constructor
     */
    PLDATrainer(const PLDATrainer& other);

    /**
     * @brief (virtual) Destructor
     */
    virtual ~PLDATrainer();

    /**
     * @brief Assignment operator
     */
    PLDATrainer& operator=(const PLDATrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const PLDATrainer& other) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const PLDATrainer& other) const;

    /**
     * @brief Similarity operator
     */
    virtual bool is_similar_to(const PLDATrainer& b, 
      const double r_epsilon=1e-5, const double a_epsilon=1e-8) const;

    /**
     * @brief Performs some initialization before the E- and M-steps.
     */
    virtual void initialization(bob::machine::PLDABase& machine, 
      const std::vector<blitz::Array<double,2> >& v_ar);
    /**
     * @brief Performs some actions after the end of the E- and M-steps.
      */
    virtual void finalization(bob::machine::PLDABase& machine, 
      const std::vector<blitz::Array<double,2> >& v_ar);
    
    /**
     * @brief Calculates and saves statistics across the dataset, and saves
     * these as m_z_{first,second}_order. 
     * The statistics will be used in the mStep() that follows.
     */
    virtual void eStep(bob::machine::PLDABase& machine, 
      const std::vector<blitz::Array<double,2> >& v_ar);

    /**
     * @brief Performs a maximization step to update the parameters of the 
     * PLDABase 
     */
    virtual void mStep(bob::machine::PLDABase& machine,
       const std::vector<blitz::Array<double,2> >& v_ar);

    /**
     * @brief Computes the average log likelihood using the current estimates
     * of the latent variables. 
     */
    virtual double computeLikelihood(bob::machine::PLDABase& machine);

    /**
     * @brief Sets whether the second order statistics are stored during the
     * training procedure, or only their sum.
     */
    void setUseSumSecondOrder(bool v) { m_use_sum_second_order = v; }
    /**
     * @brief Tells whether the second order statistics are stored during the
     * training procedure, or only their sum.
     */
    bool getUseSumSecondOrder() const 
    { return m_use_sum_second_order; }

    /**
     * @brief This enum defines different methods for initializing the \f$F\f$ 
     * subspace
     */
    typedef enum {
      RANDOM_F=0,
      BETWEEN_SCATTER=1
    }   
    InitFMethod;
    /**
     * @brief This enum defines different methods for initializing the \f$G\f$
     * subspace
     */
    typedef enum {
      RANDOM_G=0,
      WITHIN_SCATTER=1
    }   
    InitGMethod;
    /**
     * @brief This enum defines different methods for initializing the 
     * \f$\Sigma\f$ covariance matrix
     */
    typedef enum {
      RANDOM_SIGMA=0,
      VARIANCE_G=1,
      CONSTANT=2,
      VARIANCE_DATA=3
    }   
    InitSigmaMethod;
    /**
     * @brief Sets the method used to initialize \f$F\f$
     */
    void setInitFMethod(const InitFMethod m) { m_initF_method = m; }
    /**
     * @brief Gets the method used to initialize \f$F\f$
     */
    InitFMethod getInitFMethod() const { return m_initF_method; }
    /**
     * @brief Sets the ratio value used to initialize \f$F\f$
     */
    void setInitFRatio(double d) { m_initF_ratio = d; } 
    /**
     * @brief Gets the ratio value used to initialize \f$F\f$
     */
    double getInitFRatio() const { return m_initF_ratio; }
    /**
     * @brief Sets the method used to initialize \f$G\f$
     */
    void setInitGMethod(const InitGMethod m) { m_initG_method = m; }
    /**
     * @brief Gets the method used to initialize \f$G\f$
     */
    InitGMethod getInitGMethod() const { return m_initG_method; }
    /**
     * @brief Sets the ratio value used to initialize \f$G\f$ 
     */
    void setInitGRatio(double d) { m_initG_ratio = d; } 
    /**
     * @brief Gets the ratio value used to initialize \f$G\f$
     */
    double getInitGRatio() const { return m_initG_ratio; }
    /**
     * @brief Sets the method used to initialize \f$\Sigma\f$
     */
    void setInitSigmaMethod(const InitSigmaMethod m) 
    { m_initSigma_method = m; }
    /**
     * @brief Gets the method used to initialize \f$\Sigma\f$
     */
    InitSigmaMethod getInitSigmaMethod() const 
    { return m_initSigma_method; }
    /**
     * @brief Sets the ratio value used to initialize \f$\Sigma\f$
     */
    void setInitSigmaRatio(double d) { m_initSigma_ratio = d; } 
    /**
     * @brief Gets the ratio value used to initialize \f$\Sigma\f$
     */
    double getInitSigmaRatio() const { return m_initSigma_ratio; }

    /**
     * @brief Gets the z first order statistics (mostly for test purposes)
     */
    const std::vector<blitz::Array<double,2> >& getZFirstOrder() const
    { return m_cache_z_first_order;}
    /**
     * @brief Gets the z second order statistics (mostly for test purposes)
     */
    const blitz::Array<double,2>& getZSecondOrderSum() const 
    { return m_cache_sum_z_second_order;}
    /**
     * @brief Gets the z second order statistics (mostly for test purposes)
     */
    const std::vector<blitz::Array<double,3> >& getZSecondOrder() const
    { if(m_use_sum_second_order)
        throw std::runtime_error("You should disable the use_sum_second_order flag to use this feature");
      return m_cache_z_second_order;
    }

    /**
     * @brief Main procedure for enrolling a PLDAMachine
     */
    void enrol(bob::machine::PLDAMachine& plda_machine, 
      const blitz::Array<double,2>& ar) const;

  private: 
    //representation
    size_t m_dim_d; ///< Dimensionality of the input features
    size_t m_dim_f; ///< Size/rank of the \f$F\f$ subspace
    size_t m_dim_g; ///< Size/rank of the \f$G\f$ subspace
    bool m_use_sum_second_order; ///< If set, only the sum of the second order statistics is stored/allocated
    InitFMethod m_initF_method; ///< Initialization method for \f$F\f$
    double m_initF_ratio; ///< Ratio/factor used for the initialization of \f$F\f$
    InitGMethod m_initG_method; ///< Initialization method for \f$G\f$
    double m_initG_ratio; ///< Ratio/factor used for the initialization of \f$G\f$
    InitSigmaMethod m_initSigma_method; ///< Initialization method for \f$\Sigma\f$
    double m_initSigma_ratio; ///< Ratio/factor used for the initialization of \f$\Sigma\f$

    // Statistics and covariance computed during the training process
    blitz::Array<double,2> m_cache_S; ///< Covariance of the training data
    std::vector<blitz::Array<double,2> > m_cache_z_first_order; ///< Current mean of the z_{n} latent variable (1 for each sample)
    blitz::Array<double,2> m_cache_sum_z_second_order; ///< Current sum of the covariance of the z_{n} latent variable
    std::vector<blitz::Array<double,3> > m_cache_z_second_order; ///< Current covariance of the z_{n} latent variable
    // Precomputed
    /** 
     * @brief Number of training samples for each individual in the training set
     */
    std::vector<size_t> m_cache_n_samples_per_id;
    /**
     * @brief Tells if there is an identity with a 'key'/particular number of 
     * training samples, and if corresponding matrices are up to date.
     */
    std::map<size_t,bool> m_cache_n_samples_in_training;
    blitz::Array<double,2> m_cache_B; ///< \f$B = [F, G]\f$ (size nfeatures x (m_dim_f+m_dim_g) )
    blitz::Array<double,2> m_cache_Ft_isigma_G; ///< \f$F^T \Sigma^-1 G\f$
    blitz::Array<double,2> m_cache_eta; ///< \f$F^T \Sigma^-1 G \alpha\f$
    // Blocks (with \f$\gamma_{a}\f$) of \f$(Id + A^T \Sigma'^-1 A)^-1\f$ (efficient inversion)
    std::map<size_t,blitz::Array<double,2> > m_cache_zeta; ///< \f$\zeta_{a} = \alpha + \eta^T \gamma_{a} \eta\f$
    std::map<size_t,blitz::Array<double,2> > m_cache_iota; ///< \f$\iota_{a} = -\gamma_{a} \eta\f$

    // Working arrays
    mutable blitz::Array<double,1> m_tmp_nf_1; ///< vector of dimension dim_f
    mutable blitz::Array<double,1> m_tmp_nf_2; ///< vector of dimension dim_f
    mutable blitz::Array<double,1> m_tmp_ng_1; ///< vector of dimension dim_f
    mutable blitz::Array<double,1> m_tmp_D_1; ///< vector of dimension dim_d 
    mutable blitz::Array<double,1> m_tmp_D_2; ///< vector of dimension dim_d
    mutable blitz::Array<double,2> m_tmp_nfng_nfng; ///< matrix of dimension (dim_f+dim_g)x(dim_f+dim_g)
    mutable blitz::Array<double,2> m_tmp_D_nfng_1; ///< matrix of dimension (dim_d)x(dim_f+dim_g)
    mutable blitz::Array<double,2> m_tmp_D_nfng_2; ///< matrix of dimension (dim_d)x(dim_f+dim_g)

    // internal methods
    void computeMeanVariance(bob::machine::PLDABase& machine,
      const std::vector<blitz::Array<double,2> >& v_ar);
    void initMembers(const std::vector<blitz::Array<double,2> >& v_ar);
    void initFGSigma(bob::machine::PLDABase& machine, 
      const std::vector<blitz::Array<double,2> >& v_ar);
    void initF(bob::machine::PLDABase& machine, 
      const std::vector<blitz::Array<double,2> >& v_ar);
    void initG(bob::machine::PLDABase& machine, 
      const std::vector<blitz::Array<double,2> >& v_ar);
    void initSigma(bob::machine::PLDABase& machine, 
      const std::vector<blitz::Array<double,2> >& v_ar);

    void checkTrainingData(const std::vector<blitz::Array<double,2> >& v_ar);
    void precomputeFromFGSigma(bob::machine::PLDABase& machine);
    void precomputeLogLike(bob::machine::PLDABase& machine, 
      const std::vector<blitz::Array<double,2> >& v_ar);

    void updateFG(bob::machine::PLDABase& machine,
      const std::vector<blitz::Array<double,2> >& v_ar);
    void updateSigma(bob::machine::PLDABase& machine,
      const std::vector<blitz::Array<double,2> >& v_ar);

    void resizeTmp();
};

/**
 * @}
 */
}}

#endif /* BOB_TRAINER_PLDA_TRAINER_H */
