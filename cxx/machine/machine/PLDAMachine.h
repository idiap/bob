/**
 * @file cxx/machine/machine/PLDAMachine.h
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Machines that implements the Probabilistic Linear Discriminant
 *   Analysis Model of Prince and Helder,
 *   'Probabilistic Linear Discriminant Analysis for Inference About Identity',
 *   ICCV'2007
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

#ifndef BOB_MACHINE_PLDAMACHINE_H
#define BOB_MACHINE_PLDAMACHINE_H

#include <blitz/array.h>
#include "io/HDF5File.h"

namespace bob { namespace machine {
  
  /**
   * A PLDA Base machine which contains F, G and sigma matrices as well as mu.
   */
  class PLDABaseMachine {

    public:

      /**
       * Default constructor. Builds an otherwise invalid 0x0 PLDA base 
       * machine.
       */
      PLDABaseMachine();

      /**
       * Constructor, builds a new PLDA machine. F, G, sigma and mu are not 
       * initialized.
       *
       * @param d Dimensionality of the feature vector
       * @param nf size of F (d x nf)
       * @param ng size of G (d x ng)
       */ 
      PLDABaseMachine(const size_t d, const size_t nf, const size_t ng);

      /**
       * Copies another machine
       */
      PLDABaseMachine(const PLDABaseMachine& other);

      /**
       * Starts a new PLDABaseMachine from an existing Configuration object.
       */
      PLDABaseMachine(bob::io::HDF5File& config);

      /**
       * Just to virtualize the destructor
       */
      virtual ~PLDABaseMachine(); 

      /**
       * Assigns from a different machine
       */
      PLDABaseMachine& operator= (const PLDABaseMachine &other);

      /**
       * Loads data from an existing configuration object. Resets the current
       * state.
       */
      void load(bob::io::HDF5File& config);

      /**
       * Saves an existing machine to a Configuration object.
       */
      void save(bob::io::HDF5File& config) const;

      /** 
       * Resizes the PLDA Machine. F, G, sigma and mu will should be 
       * considered uninitialized.
       */
      void resize(const size_t d, const size_t nf, const size_t ng);

      /**
        * Gets the F matrix
        */
      inline const blitz::Array<double,2>& getF() const 
      { return m_F; }

      /**
        * Sets the F matrix
        */
      void setF(const blitz::Array<double,2>& F);

      /**
       * Returns the current F matrix in order to be updated.
       * @warning Use with care. Only trainers should use this function for
       * efficiency reasons.
       */
      inline blitz::Array<double, 2>& updateF()  
      { return m_F; }

      /**
        * Gets the G matrix
        */
      inline const blitz::Array<double,2>& getG() const 
      { return m_G; }

      /**
        * Sets the G matrix
        */
      void setG(const blitz::Array<double,2>& G);

      /**
       * Returns the current G matrix in order to be updated.
       * @warning Use with care. Only trainers should use this function for
       * efficiency reasons.
       */
      inline blitz::Array<double, 2>& updateG()
      { return m_G; }

      /**
        * Gets the sigma (diagonal) 'matrix'
        */
      inline const blitz::Array<double,1>& getSigma() const 
      { return m_sigma; }

      /**
        * Sets the sigma matrix
        */
      void setSigma(const blitz::Array<double,1>& s);

      /**
       * Returns the current sigma matrix in order to be updated.
       * @warning Use with care. Only trainers should use this function for
       * efficiency reasons.
       */
      inline blitz::Array<double, 1>& updateSigma()
      { return m_sigma; }

      /**
        * Gets the mu vector
        */
      inline const blitz::Array<double,1>& getMu() const 
      { return m_mu; }

      /**
        * Sets the mu vector
        */
      void setMu(const blitz::Array<double,1>& mu);

      /**
       * Returns the current mu vector in order to be updated.
       * @warning Use with care. Only trainers should use this function for
       * efficiency reasons.
       */
      inline blitz::Array<double, 1>& updateMu()
      { return m_mu; }


      /**
        * Gets the feature dimensionality
        */
      inline size_t getDimD() const 
      { return m_mu.extent(0); }

      /**
        * Gets the size/rank the F matrix along the second dimension
        */
      inline size_t getDimF() const 
      { return m_F.extent(1); }

      /**
        * Gets the size/rank the G matrix along the second dimension
        */
      inline size_t getDimG() const 
      { return m_G.extent(1); }

      /**
        * Precomputes the useful values alpha, beta and gamma
        */
      void precompute();
      /**
        * Precomputes useful values for the log likelihood
        */
      void precomputeLogLike();
      /**
        * Gets the inverse vector/diagonal matrix of sigma
        * isigma = sigma^-1
        */
      inline const blitz::Array<double,1>& getISigma() const 
      { return m_isigma; }
      /**
       * Returns the inverse vector/diagonal matrix of sigma.
       * @warning Use with care. Only trainers should use this function.
       */
      inline blitz::Array<double, 1>& updateISigma()
      { return m_isigma; }
      /**
        * Gets the alpha matrix.
        * alpha = (Id + G^T.sigma^-1.G)^-1
        */
      inline const blitz::Array<double,2>& getAlpha() const 
      { return m_alpha; }
      /**
       * Returns the alpha matrix.
       * @warning Use with care. Only trainers should use this function.
       */
      inline blitz::Array<double, 2>& updateAlpha()
      { return m_alpha; }
      /**
        * Gets the beta matrix
        * beta = (sigma + G.G^T)^-1
        */
      inline const blitz::Array<double,2>& getBeta() const 
      { return m_beta; }
      /**
       * Returns the beta matrix.
       * @warning Use with care. Only trainers should use this function.
       */
      inline blitz::Array<double, 2>& updateBeta()
      { return m_beta; }
      /**
        * Gets the gamma matrix for a given a (number of samples)
        * gamma_a = (Id + a.F^T.beta.F)^-1
        * @warning an exception is thrown if gamma does not exists
        */
      blitz::Array<double,2>& getGamma(const size_t a);
      /**
        * Gets the gamma matrix for a given a (number of samples)
        * gamma_a = (Id + a.F^T.beta.F)^-1
        * @warning The matrix is computed if it does not already exists
        */
      blitz::Array<double,2>& getAddGamma(const size_t a);
      /**
        * Gets the Ft.beta matrix
        * F^t.beta = Ft.(sigma + G.G^T)^-1
        */
      inline const blitz::Array<double,2>& getFtBeta() const 
      { return m_Ft_beta; }
      /**
       * Returns the Ft.beta matrix.
       * @warning Use with care. Only trainers should use this function.
       */
      inline blitz::Array<double, 2>& updateFtBeta()
      { return m_Ft_beta; }
      /**
        * Gets the Gt.sigma^-1 matrix
        */
      inline const blitz::Array<double,2>& getGtISigma() const 
      { return m_Gt_isigma; }
      /**
       * Returns the Gt.sigma^-1 matrix.
       * @warning Use with care. Only trainers should use this function.
       */
      inline blitz::Array<double, 2>& updateGtISigma()
      { return m_Gt_isigma; }
      /**
        * Gets log(det(alpha)) 
        */
      inline const double getLogDetAlpha() const 
      { return m_logdet_alpha; }
      /**
        * Gets log(det(sigma)) 
        */
      inline const double getLogDetSigma() const 
      { return m_logdet_sigma; }
      /**
        * Computes the log likelihood constant term for a given a 
        * (number of samples), given the provided gamma_a matrix
        * loglike_constterm[a] = a/2 * 
        *   ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
        */
      double computeLogLikeConstTerm(const size_t a, 
        const blitz::Array<double,2>& gamma_a);
      /**
        * Computes the log likelihood constant term for a given a 
        * (number of samples)
        * loglike_constterm[a] = a/2 * 
        *   ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
        * @warning: gamma_a will be computed and added if it does
        *  not already exists
        */
      double computeLogLikeConstTerm(const size_t a);
      /**
        * Tells if the log likelihood constant term for a given a 
        * (number of samples) exists
        * loglike_constterm[a] = a/2 * 
        *   ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
        */
      inline bool hasLogLikeConstTerm(const size_t a) const
      { return (m_loglike_constterm.find(a) != m_loglike_constterm.end()); }
      /**
        * Gets the log likelihood constant term for a given a \
        * (number of samples)
        * loglike_constterm[a] = a/2 * 
        *   ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
        * @warning an exception is thrown if the value does not exists
        */
      double getLogLikeConstTerm(const size_t a);
      /**
        * Gets the log likelihood constant term for a given a \
        * (number of samples)
        * loglike_constterm[a] = a/2 * 
        *   ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
        * @warning The value is computed if it does not already exists
        */
      double getAddLogLikeConstTerm(const size_t a);

      /**
        * Computes the gamma matrix for a given a (number of samples)
        * and put the result in res.
        * gamma_a = (Id + a.F^T.beta.F)^-1
        */
      void computeGamma(const size_t a, blitz::Array<double,2> res);
      /**
        * Tells if the gamma matrix for a given a (number of samples) exists
        * gamma_a = (Id + a.F^T.beta.F)^-1
        */
      inline bool hasGamma(const size_t a) const
      { return (m_gamma.find(a) != m_gamma.end()); }
      /**
        * Clears the maps (gamma_a and m_loglike_constterm).
        */
      void clearMaps();

    private:
      // F, G and sigma matrices, and mu vector
      // sigma is assumed to be diagonal, and only the diagonal is stored
      blitz::Array<double,2> m_F;
      blitz::Array<double,2> m_G;
      blitz::Array<double,1> m_sigma; 
      blitz::Array<double,1> m_mu; 

      // Internal values very useful used to optimize the code
      // isigma = sigma^-1
      blitz::Array<double,1> m_isigma; 
      // alpha = (Id + G^T.sigma^-1.G)^-1
      blitz::Array<double,2> m_alpha;
      // beta = (sigma+G.G^T)^-1 = (sigma^-1 - sigma^-1.G.alpha.G^T.sigma^-1)^-1
      blitz::Array<double,2> m_beta;
      // gamma_a = (Id + a.F^T.beta.F)^-1 (depends on the number of samples)
      std::map<size_t, blitz::Array<double,2> > m_gamma;
      // Ft_beta = F^T.(sigma+G.G^T)^-1 = F^T.(sigma^-1 - sigma^-1.G.alpha.G^T.sigma^-1)^-1
      blitz::Array<double,2> m_Ft_beta;
      // Gt_isigma = G^T.sigma^-1
      blitz::Array<double,2> m_Gt_isigma;
      // log(det(alpha)) and log(det(sigma))
      double m_logdet_alpha;
      double m_logdet_sigma;
      // Log likelihood constant term which depends on the number of samples a
      // loglike_constterm[a] = a/2 * 
      //    ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
      std::map<size_t, double> m_loglike_constterm;

      // cache
      blitz::Array<double,2> m_cache_d_ng_1;
      blitz::Array<double,2> m_cache_nf_nf_1;
      blitz::Array<double,2> m_cache_ng_ng_1;

      void initFGSigma();
      void precomputeISigma();
      void precomputeAlpha();
      void precomputeBeta();
      void precomputeGamma(const size_t a);
      void precomputeFtBeta();
      void precomputeGtISigma();
      void precomputeLogDetAlpha();
      void precomputeLogDetSigma();
      void precomputeLogLikeConstTerm(const size_t a);
  };


  /**
   * A PLDA machine which contains elements from enrollment samples.
   */
  class PLDAMachine {

    public:

      /**
       * Default constructor. Builds an otherwise invalid 0 x 0 PLDA machine.
       */
      PLDAMachine();

      /**
       * Constructor, builds a new PLDA machine, setting a PLDABaseMachine. 
       */ 
      PLDAMachine(const boost::shared_ptr<bob::machine::PLDABaseMachine> pldabase);

      /**
       * Copies another machine
       */
      PLDAMachine(const PLDAMachine& other);

      /**
       * Starts a new PLDAMachine from an existing Configuration object.
       */
      PLDAMachine(bob::io::HDF5File& config);

      /**
       * Just to virtualise the destructor
       */
      virtual ~PLDAMachine(); 

      /**
       * Assigns from a different machine
       */
      PLDAMachine& operator= (const PLDAMachine &other);

      /**
       * Loads data from an existing configuration object. Resets the current
       * state.
       */
      void load(bob::io::HDF5File& config);

      /** 
       * Resizes the PLDA Machine.
       */
      void resize(const size_t d, const size_t nf, const size_t ng);

      /**
       * Saves an existing machine to a Configuration object.
       */
      void save(bob::io::HDF5File& config) const;

      /**
        * Get the PLDABaseMachine
        */
      const boost::shared_ptr<bob::machine::PLDABaseMachine> getPLDABase() const 
      { return m_plda_base; }

      /**
        * Gets the feature dimensionality
        */
      inline size_t getDimD() const 
      { return m_plda_base->getDimD(); }

      /**
        * Gets the size/rank the F matrix along the second dimension
        */
      inline size_t getDimF() const 
      { return m_plda_base->getDimF(); }

      /**
        * Gets the size/rank the G matrix along the second dimension
        */
      inline size_t getDimG() const 
      { return m_plda_base->getDimG(); }


      /**
        * Gets the number of enrolled samples
        */
      inline uint64_t getNSamples() const
      { return m_n_samples; }
      /**
        * Sets the number of enrolled samples
        */
      void setNSamples(const uint64_t n_samples)
      { m_n_samples = n_samples; }
      /**
        * Gets the nh_sum_xit_beta_xi value
        */
      inline double getWSumXitBetaXi() const
      { return m_nh_sum_xit_beta_xi; }
      /**
        * Sets the nh_sum_xit_beta_xi value
        */
      void setWSumXitBetaXi(const double val)
      { m_nh_sum_xit_beta_xi = val; }
      /**
       * Gets the current weighted sum
       */
      inline const blitz::Array<double, 1>& getWeightedSum()
      { return m_weighted_sum; }
      /**
        * Set the Weigted sum
        */
      void setWeightedSum(const blitz::Array<double,1>& weighted_sum);
      /**
       * Returns the current weighted sum in order to be updated.
       * @warning Use with care. Only trainers should use this function for
       * efficiency reasons.
       */
      inline blitz::Array<double, 1>& updateWeightedSum()
      { return m_weighted_sum; }
      /**
        * Gets the log likelihood of the enrollment samples
        */
      inline double getLogLikelihood() const
      { return m_loglikelihood; }
      /**
        * Sets the log likelihood of the enrollment samples
        */
      void setLogLikelihood(const double val)
      { m_loglikelihood = val; }

      /**
        * Set the PLDABaseMachine
        */
      void setPLDABase(const boost::shared_ptr<bob::machine::PLDABaseMachine> plda_base);


      /**
        * Tells if the gamma matrix for a given a (number of samples) exists
        * in this machine (does not check the base machine)
        * gamma_a = (Id + a.F^T.beta.F)^-1
        */
      inline bool hasGamma(const size_t a) const
      { return (m_gamma.find(a) != m_gamma.end()); }
      /**
        * Gets the gamma matrix for a given a (number of samples)
        * gamma_a = (Id + a.F^T.beta.F)^-1
        * Tries to find it from the base machine and then from this machine
        * @warning an exception is thrown if gamma does not exists
        */
      blitz::Array<double,2>& getGamma(const size_t a);
      /**
        * Gets the gamma matrix for a given a (number of samples)
        * gamma_a = (Id + a.F^T.beta.F)^-1
        * Tries to find it from the base machine and then from this machine
        * @warning The matrix is computed if it does not already exists,
        *   and stored in this machine
        */
      blitz::Array<double,2>& getAddGamma(const size_t a);
      /**
        * Tells if the log likelihood constant term for a given a 
        * (number of samples) exists in this machine 
        * (does not check the base machine)
        * loglike_constterm[a] = a/2 * 
        *   ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
        */
      inline bool hasLogLikeConstTerm(const size_t a) const
      { return (m_loglike_constterm.find(a) != m_loglike_constterm.end()); }
      /**
        * Gets the log likelihood constant term for a given a \
        * (number of samples)
        * Tries to find it from the base machine and then from this machine
        * loglike_constterm[a] = a/2 * 
        *   ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
        * @warning an exception is thrown if the value does not exists
        */
      double getLogLikeConstTerm(const size_t a);
      /**
        * Gets the log likelihood constant term for a given a \
        * (number of samples)
        * Tries to find it from the base machine and then from this machine
        * loglike_constterm[a] = a/2 * 
        *   ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
        * @warning The value is computed if it does not already exists
        */
      double getAddLogLikeConstTerm(const size_t a);
      /**
        * Clears the maps (gamma_a and m_loglike_constterm).
        */
      void clearMaps();


      /**
        * Compute the likelihood of the given sample and (optionnaly) 
        * the enrolled samples
        */
      double computeLikelihood(const blitz::Array<double,1>& sample,
        bool with_enrolled_samples=true);
      /**
        * Compute the likelihood of the given samples and (optionnaly) 
        * the enrolled samples
        */
      double computeLikelihood(const blitz::Array<double,2>& samples,
        bool with_enrolled_samples=true);

      /**
        * Computes a log likelihood ratio from a 1D or 2D blitz::Array
        */
      void forward(const blitz::Array<double,1>& sample, double& score);
      void forward(const blitz::Array<double,2>& samples, double& score);


    private:
      /**
        * Base PLDA Machine containing the model (F, G and sigma)
        */
      boost::shared_ptr<bob::machine::PLDABaseMachine> m_plda_base;

      /**
        * Number of enrollement samples
        */
      uint64_t m_n_samples;
      /**
        * Contains the value:
        * A = -0.5 sum_i xi^T.sigma^-1.xi - xi^T.sigma^-1.G.alpha.G^T.sigma^-1.x_i
        * A = -0.5 sum_i xi^T.beta.x_i
        * used in the likelihood computation (first xi dependent term)
        */
      double m_nh_sum_xit_beta_xi;
      /**
        * Contains the value sum_i F^T.beta.xi
        * used in the likelihood computation (for the second xi dependent term)
        */
      blitz::Array<double,1> m_weighted_sum;
      /**
        * Log likelihood of the enrolled samples
        */
      double m_loglikelihood;

      // Values which are not already in the base machine
      // gamma_a = (Id + a.F^T.beta.F)^-1 (depends on the number of samples)
      std::map<size_t, blitz::Array<double,2> > m_gamma;
      // Log likelihood constant term which depends on the number of samples a
      // loglike_constterm[a] = a/2 * 
      //    ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
      std::map<size_t, double> m_loglike_constterm;


      // cache
      blitz::Array<double,1> m_cache_d_1;
      blitz::Array<double,1> m_cache_d_2;
      blitz::Array<double,1> m_cache_nf_1;
      blitz::Array<double,1> m_cache_nf_2;
  };


}}

#endif
