/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 06 Oct 2011
 *
 * @brief Base class for Expectation-Maximization-like algorithms
 */


#ifndef TORCH5SPRO_TRAINER_EMTRAINERNEW_H
#define TORCH5SPRO_TRAINER_EMTRAINERNEW_H

#include "trainer/Trainer.h"

#include <limits>
#include "core/logging.h"


namespace Torch { namespace trainer {
  
  /**
    * @brief This class implements the general Expectation-maximization algorithm.
    * @details See Section 9.3 of Bishop, "Pattern recognition and machine learning", 2006
    * Derived classes must implement the initialization(), eStep(), mStep() and finalization() methods.
    */
  template<class T_machine, class T_sampler>
  class EMTrainerNew: virtual public Trainer<T_machine, T_sampler>
  {
  public:
    virtual ~EMTrainerNew() {}
    
    virtual void train(T_machine& machine, const T_sampler& sampler) 
    {
      Torch::core::info << "# EMTrainerNew:" << std::endl;
      
      /*
      // Check that the machine and dataset have the same feature dimensionality
      if (!checkForDimensionalityMatch()) 
      {
        Torch::core::error << "Mismatch in dimensionality of dataset and machine" << endl;
        return false;
      }
      */
      
      // Initialization
      initialization(machine, sampler);
      // Do the Expectation-Maximization algorithm
      double average_output_previous = - std::numeric_limits<double>::max();
      double average_output = - std::numeric_limits<double>::max();
      
      // - iterates...
      for(size_t iter=0; ; ++iter) {
        
        // - saves average output from last iteration
        average_output_previous = average_output;
       
        // - eStep
        eStep(machine, sampler);
   
        // - mStep
        mStep(machine, sampler);
        
        // - Computes log likelihood if required
        if(m_compute_likelihood) {
          average_output = computeLikelihood(machine, sampler);
        
          Torch::core::info << "# Iteration " << iter+1 << ": " 
            << average_output_previous << " -> " 
            << average_output << std::endl;
        
          // - Terminates if converged (and likelihood computation is set)
          if(fabs((average_output_previous - average_output)/average_output_previous) <= m_convergence_threshold) {
            Torch::core::info << "# EM terminated: likelihood converged" << std::endl;
            break;
          }
        }
        else
          Torch::core::info << "# Iteration " << iter+1 << std::endl;
        
        // - Terminates if maximum number of iterations has been reached
        if(m_max_iterations > 0 && iter+1 >= m_max_iterations) {
          Torch::core::info << "# EM terminated: maximum number of iterations reached." << std::endl;
          break;
        }
      }

      // Finalization
      finalization(machine, sampler);
    }

    /**
      * This method is called before the EM algorithm 
      */
    virtual void initialization(T_machine& machine, const T_sampler& sampler) = 0;
    
    /**
      * Updates the hidden variable distribution (or the sufficient statistics)
      * given the Machine parameters.
      */
    virtual void eStep(T_machine& machine, const T_sampler& sampler) = 0;
    
    /**
      * Update the Machine parameters given the hidden variable distribution 
      * (or the sufficient statistics)
      */
    virtual void mStep(T_machine& machine, const T_sampler& sampler) = 0;

    /**
      * @return The average output of the Machine across the dataset.
      *     The EM algorithm will terminate once the change in average_output
      *     is less than the convergence_threshold.
      */
    virtual double computeLikelihood(T_machine& machine, const T_sampler& sampler) = 0;

    /**
      * This method is called after the EM algorithm 
      */
    virtual void finalization(T_machine& machine, const T_sampler& sampler) = 0;

    /**
      * Sets likelihood computation
      */
    void setComputeLikelihood(bool compute) {
      m_compute_likelihood = compute;
    }

    /**
      * Gets convergence threshold
      */
    bool getComputeLikelihood() const {
      return m_compute_likelihood;
    }

    /**
      * Sets convergence threshold
      */
    void setConvergenceThreshold(double threshold) {
      m_convergence_threshold = threshold;
    }

    /**
      * Gets convergence threshold
      */
    double getConvergenceThreshold() const {
      return m_convergence_threshold;
    }

    /**
      * Set max iterations
      */
    void setMaxIterations(size_t max_iterations) {
      m_max_iterations = max_iterations;
    }

    /**
      * Get max iterations
      */
    size_t getMaxIterations() const {
      return m_max_iterations;
    }

  protected:
    bool m_compute_likelihood;
    double m_convergence_threshold;
    size_t m_max_iterations;

    /**
      * Protected constructor to be called in the constructor of derived 
      * classes
      */
    EMTrainerNew(double convergence_threshold = 0.001, 
        size_t max_iterations = 10, bool compute_likelihood = true):
      m_compute_likelihood(compute_likelihood), 
      m_convergence_threshold(convergence_threshold), 
      m_max_iterations(max_iterations) 
    {
    }
  };

}}

#endif // TORCH5SPRO_TRAINER_EMTRAINERNEW_H
