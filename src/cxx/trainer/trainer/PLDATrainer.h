/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue 11 Oct 2011
 *
 * @brief Probabilistic PLDA Discriminant Analysis implemented using 
 * Expectation Maximization.
 */

#ifndef TORCH5SPRO_TRAINER_PLDA_TRAINER_H
#define TORCH5SPRO_TRAINER_PLDA_TRAINER_H

#include <blitz/array.h>
#include <map>
#include <vector>
#include "trainer/EMTrainerNew.h"
#include "machine/PLDAMachine.h"
#include "io/Arrayset.h"

namespace Torch { namespace trainer {
  
  /**
    */
  class PLDABaseTrainer: public EMTrainerNew<Torch::machine::PLDABaseMachine, 
                                             std::vector<Torch::io::Arrayset> >
  {
    public: //api
      /**
        * Initializes a new PLDA trainer. The training stage will place the
        * resulting components in the PLDABaseMachine and set it up to
        * extract the variable means automatically. 
        */
      PLDABaseTrainer(int nf, int ng, double convergence_threshold=0.001,
        int max_iterations=10, bool compute_likelihood=false);

      /**
        * Copy construction.
        */
      PLDABaseTrainer(const PLDABaseTrainer& other);

      /**
        * (virtual) Destructor
        */
      virtual ~PLDABaseTrainer();

      /**
        * Copy operator
        */
      PLDABaseTrainer& operator=(const PLDABaseTrainer& other);

      /**
        * This methods performs some initialization before the E- and M-steps.
        */
      virtual void initialization(Torch::machine::PLDABaseMachine& machine, 
        const std::vector<Torch::io::Arrayset>& v_ar);
      /**
        * This methods performs some actions after the end of the E- and 
        * M-steps.
        */
      virtual void finalization(Torch::machine::PLDABaseMachine& machine, 
        const std::vector<Torch::io::Arrayset>& v_ar);
      
      /**
        * Calculates and saves statistics across the dataset, and saves these 
        * as m_z_{first,second}_order. 
        * 
        * The statistics will be used in the mStep() that follows.
        */
      virtual void eStep(Torch::machine::PLDABaseMachine& machine, 
        const std::vector<Torch::io::Arrayset>& v_ar);

      /**
        * Performs a maximization step to update the parameters of the 
        */
      virtual void mStep(Torch::machine::PLDABaseMachine& machine,
         const std::vector<Torch::io::Arrayset>& v_ar);

      /**
        * Computes the average log likelihood using the current estimates of 
        * the latent variables. 
        */
      virtual double computeLikelihood(Torch::machine::PLDABaseMachine& machine,
         const std::vector<Torch::io::Arrayset>& v_ar);

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
        * Sets the method used to initialize F
        */
      inline void setInitFMethod(int m) { m_initF_method = m; } 
      /**
        * Gets the method used to initialize F
        */
      inline int getInitFMethod() const { return m_initF_method; }
      /**
        * Sets the ratio value used to initialize F
        */
      inline void setInitFRatio(double d) { m_initF_ratio = d; } 
      /**
        * Gets the ratio value used to initialize F
        */
      inline double getInitFRatio() const { return m_initF_ratio; }

      /**
        * Sets the method used to initialize G
        */
      inline void setInitGMethod(int m) { m_initG_method = m; } 
      /**
        * Gets the method used to initialize G
        */
      inline int getInitGMethod() const { return m_initG_method; }
      /**
        * Sets the ratio value used to initialize G
        */
      inline void setInitGRatio(double d) { m_initG_ratio = d; } 
      /**
        * Gets the ratio value used to initialize G
        */
      inline double getInitGRatio() const { return m_initG_ratio; }

      /**
        * Sets the method used to initialize sigma
        */
      inline void setInitSigmaMethod(int m) { m_initSigma_method = m; } 
      /**
        * Gets the method used to initialize sigma
        */
      inline int getInitSigmaMethod() const { return m_initSigma_method; }
      /**
        * Sets the ratio value used to initialize sigma
        */
      inline void setInitSigmaRatio(double d) { m_initSigma_ratio = d; } 
      /**
        * Gets the ratio value used to initialize sigma
        */
      inline double getInitSigmaRatio() const { return m_initSigma_ratio; }





      /**
        * Gets the z first order statistics (mostly for test purposes)
        */
      inline const std::vector<blitz::Array<double,2> >& getZFirstOrder() const
      { return m_z_first_order;}
      /**
        * Gets the z second order statistics (mostly for test purposes)
        */
      inline const blitz::Array<double,2>& getZSecondOrderSum() const 
      { return m_sum_z_second_order;}

    private: 
      //representation
      int m_nf;
      int m_ng;
      blitz::Array<double,2> m_S; /// Covariance of the training data
      std::vector<blitz::Array<double,2> > m_z_first_order; /// Current mean of the z_{n} latent variable (1 for each sample)
      blitz::Array<double,2> m_sum_z_second_order; /// Current sum of the covariance of the z_{n} latent variable
      int m_seed; /// The seed for the random initialization of W and sigma2
      int m_initF_method;
      double m_initF_ratio;
      int m_initG_method;
      double m_initG_ratio;
      int m_initSigma_method;
      double m_initSigma_ratio;

      std::vector<blitz::Array<double,1> > m_y_first_order; /// Current mean of the y_{n} latent variable
      std::vector<blitz::Array<double,2> > m_y_second_order; /// Current covariance of the y_{n} latent variable

      // Precomputed

      // Number of training samples for each individual in the training set
      std::vector<size_t> m_n_samples_per_id;
      // Tells if there is an identity with a 'key'/particular number of 
      // training samples, and if corresponding matrices are up to date
      std::map<size_t,bool> m_n_samples_in_training;

      blitz::Array<double,2> m_B; /// B = [F G] (size nfeatures x (m_nf+m_ng) )
      blitz::Array<double,2> m_Ft_isigma_G; // F^T.sigma^-1.G
      blitz::Array<double,2> m_eta; // F^T.sigma^-1.G.alpha
      // Blocks (with gamma_a) of (Id + A^T.sigma'^-1.A)^-1 (efficient inversion)
      std::map<size_t,blitz::Array<double,2> > m_zeta; // zeta_a = alpha + eta^T.gamma_a.eta
      std::map<size_t,blitz::Array<double,2> > m_iota; // iota_a = - gamma_a.eta

      // Cache
      mutable blitz::Array<double,1> m_cache_nf;
      mutable blitz::Array<double,1> m_cache_D_1; // D=nb features
      mutable blitz::Array<double,1> m_cache_D_2; // D=nb features
      mutable std::map<size_t,blitz::Array<double,1> > m_cache_for_y_first_order;
      mutable blitz::Array<double,2> m_cache_nfng_nfng;
      mutable blitz::Array<double,2> m_cache_D_nfng_1; // D=nb features, nfng=nf+ng
      mutable blitz::Array<double,2> m_cache_D_nfng_2; // D=nb features, nfng=nf+ng

      // internal methods
      void computeMeanVariance(Torch::machine::PLDABaseMachine& machine,
        const std::vector<Torch::io::Arrayset>& v_ar);
      void initMembers(const std::vector<Torch::io::Arrayset>& v_ar);
      void initFGSigma(Torch::machine::PLDABaseMachine& machine, 
        const std::vector<Torch::io::Arrayset>& v_ar);
      void initF(Torch::machine::PLDABaseMachine& machine, 
        const std::vector<Torch::io::Arrayset>& v_ar);
      void initG(Torch::machine::PLDABaseMachine& machine, 
        const std::vector<Torch::io::Arrayset>& v_ar);
      void initSigma(Torch::machine::PLDABaseMachine& machine, 
        const std::vector<Torch::io::Arrayset>& v_ar);
      void initRandomFGSigma(Torch::machine::PLDABaseMachine& machine);

      void checkTrainingData(const std::vector<Torch::io::Arrayset>& v_ar);
      void precomputeFromFGSigma(Torch::machine::PLDABaseMachine& machine);
      void precomputeLogLike(Torch::machine::PLDABaseMachine& machine, 
        const std::vector<Torch::io::Arrayset>& v_ar);

      void updateFG(Torch::machine::PLDABaseMachine& machine,
        const std::vector<Torch::io::Arrayset>& v_ar);
      void updateSigma(Torch::machine::PLDABaseMachine& machine,
        const std::vector<Torch::io::Arrayset>& v_ar);
  };



  class PLDATrainer {

    public:
      /**
       * Initializes a new PLDA trainer.
       */
      PLDATrainer(Torch::machine::PLDAMachine& plda_machine);

      /**
        * Copy constructor.
        */
      PLDATrainer(const PLDATrainer& other);

      /**
       * Destructor virtualisation
       */
      virtual ~PLDATrainer();

      /**
        * Copy operator
        */
      PLDATrainer& operator=(const PLDATrainer& other);

      /**
        * Main procedure for enrolling with this PLDA trainer
        */
      void enrol(const Torch::io::Arrayset& ar);

    private:

      Torch::machine::PLDAMachine& m_plda_machine; // PLDAMachine

      // cache
      mutable blitz::Array<double,1> m_cache_D_1; // D=nb features 
      mutable blitz::Array<double,1> m_cache_D_2; // D=nb features 
      mutable blitz::Array<double,1> m_cache_nf_1; // nf 
  };

}}

#endif /* TORCH5SPRO_TRAINER_PLDA_TRAINER_H */
