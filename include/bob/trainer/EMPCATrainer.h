/**
 * @file bob/trainer/EMPCATrainer.h
 * @date Tue Oct 11 12:18:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Expectation Maximization Algorithm for Principal Component 
 * Analysis
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

#ifndef BOB_TRAINER_EMPCA_TRAINER_H
#define BOB_TRAINER_EMPCA_TRAINER_H

#include <blitz/array.h>
#include "bob/trainer/EMTrainer.h"
#include "bob/machine/LinearMachine.h"

namespace bob { namespace trainer {
  
  /**
    * Sets a linear machine to perform Expectation Maximization on a
    * given dataset. References:
    *  1. "Probabilistic Principal Component Analysis", 
    *     Michael Tipping and Christopher Bishop,
    *     Journal of the Royal Statistical Society,
    *      Series B, 61, Part 3, pp. 611–622
    *  2. "EM Algorithms for PCA and SPCA", 
    *     Sam Roweis, Neural Information Processing Systems 10 (NIPS'97), 
    *     pp.626-632 (Sensible Principal Component Analysis part)
    *
    * Notations used are the ones from reference 1.
    * The probabilistic model is given by: t = W x + mu + epsilon
    *  - t is the observed data (dimension f)
    *  - W is a projection matrix (dimension f x d)
    *  - x is the projected data (dimension d < f)
    *  - mu is the mean of the data (dimension f)
    *  - epsilon is the noise of the data (dimension f)
    *      Gaussian with zero-mean and covariance matrix sigma^2 * Id
    */
  class EMPCATrainer: public EMTrainer<bob::machine::LinearMachine, 
                                          blitz::Array<double,2> > 
  {
    public: //api
      /**
        * Initializes a new EM PCA trainer. The training stage will place the
        * resulting components in the linear machine and set it up to
        * extract the variable means automatically. 
        */
      EMPCATrainer(int dimensionality, double convergence_threshold=0.001, 
        int max_iterations=10, bool compute_likelihood=true); 

      /**
        * Copy constructor
        */
      EMPCATrainer(const EMPCATrainer& other);

      /**
        * (virtual) Destructor
        */
      virtual ~EMPCATrainer();

      /**
        * Copy operator
        */
      EMPCATrainer& operator=(const EMPCATrainer& other);

      /**
        * This methods performs some initialization before the E- and M-steps.
        */
      virtual void initialization(bob::machine::LinearMachine& machine, 
        const blitz::Array<double,2>& ar);
      /**
        * This methods performs some actions after the end of the E- and 
        * M-steps.
        */
      virtual void finalization(bob::machine::LinearMachine& machine, 
        const blitz::Array<double,2>& ar);
      
      /**
        * Calculates and saves statistics across the dataset, and saves these 
        * as m_z_{first,second}_order. 
        * 
        * The statistics will be used in the mStep() that follows.
        */
      virtual void eStep(bob::machine::LinearMachine& machine, 
        const blitz::Array<double,2>& ar);

      /**
        * Performs a maximization step to update the parameters of the 
        */
      virtual void mStep(bob::machine::LinearMachine& machine,
         const blitz::Array<double,2>& ar);

      /**
        * Computes the average log likelihood using the current estimates of 
        * the latent variables. 
        */
      virtual double computeLikelihood(bob::machine::LinearMachine& machine);

      /**
        * Sets the seed used to generate pseudo-random numbers 
        * Used to initialize sigma2 and the projection matrix W
        */
      inline void setSeed(int seed) { m_seed = seed; }

      /**
        * Gets the seed
        */
      inline int getSeed() const { return m_seed; }

      /**
        * Sets sigma2 (Mostly for test purpose)
        */
      inline void setSigma2(double sigma2) { m_sigma2 = sigma2; }

      /**
        * Gets sigma2 (Mostly for test purpose)
        */
      inline double getSigma2() const { return m_sigma2; }

    private: //representation
      double m_dimensionality; /// Dimensionality of the new/projected data
      blitz::Array<double,2> m_S; /// Covariance of the training data (required only if we need to compute the log likelihood)
      blitz::Array<double,2> m_z_first_order; /// Current mean of the z_{n} latent variable
      blitz::Array<double,3> m_z_second_order; /// Current covariance of the z_{n} latent variable
      blitz::Array<double,2> m_inW; /// The matrix product W^T.W
      blitz::Array<double,2> m_invM; /// The matrix inv(M), where M = W^T.W + sigma2*Id
      double m_sigma2; /// The variance sigma^2 of the noise epsilon of the probabilistic model
      double m_f_log2pi; /// The constant n_features * log(2*PI) used during the likelihood computation
      int m_seed; /// The seed for the random initialization of W and sigma2

      // Cache
      mutable blitz::Array<double,2> m_cache_dxf; /// Cache of size m_dimensionality x n_features
      mutable blitz::Array<double,1> m_cache_d; /// Cache of size m_dimensionality
      mutable blitz::Array<double,1> m_cache_f; /// Cache of size n_features
      mutable blitz::Array<double,2> m_cache_dxd_1; /// Cache of size m_dimensionality x m_dimensionality
      mutable blitz::Array<double,2> m_cache_dxd_2; /// Cache of size m_dimensionality x m_dimensionality
      mutable blitz::Array<double,2> m_cache_fxd_1; /// Cache of size n_features x m_dimensionality 
      mutable blitz::Array<double,2> m_cache_fxd_2; /// Cache of size n_features x m_dimensionality 
      mutable blitz::Array<double,2> m_cache_fxf_1; /// Cache of size n_features x n_features
      mutable blitz::Array<double,2> m_cache_fxf_2; /// Cache of size n_features x n_features


      /**
        * Initializes/resizes the (array) members
        */
      void initMembers(const blitz::Array<double,2>& ar);
      /**
        * Computes the mean and the variance (if required) of the training data
        */
      void computeMeanVariance(bob::machine::LinearMachine& machine, 
        const blitz::Array<double,2>& ar);
      /**
        * Random initialization of W and sigma2
        * W is the projection matrix (from the LinearMachine)
        */
      void initRandomWSigma2(bob::machine::LinearMachine& machine);
      /**
        * Computes the product W^T.W
        * W is the projection matrix (from the LinearMachine)
        */
      void computeWtW(bob::machine::LinearMachine& machine);
      /**
        * Computes the inverse of M matrix, where M = W^T.W + sigma2.Id
        * W is the projection matrix (from the LinearMachine)
        */
      void computeInvM();
      /**
        * M-Step (part 1): Computes the new estimate of W using the new 
        * estimated statistics
        */
      void updateW(bob::machine::LinearMachine& machine,
         const blitz::Array<double,2>& ar);
      /**
        * M-Step (part 2): Computes the new estimate of sigma2 using the new 
        * estimated statistics
        */
      void updateSigma2(bob::machine::LinearMachine& machine,
         const blitz::Array<double,2>& ar);
  };

}}

#endif /* BOB_TRAINER_EMPCA_TRAINER_H */
