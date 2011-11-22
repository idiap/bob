/**
 * @file cxx/trainer/trainer/EMTrainer.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#ifndef EMTRAINER_H
#define EMTRAINER_H

#include "Trainer.h"

#include <cfloat>
#include <core/logging.h>


namespace Torch {
namespace trainer {
  
/// @brief This class implements the general expectation-maximisation algorithm.
/// @details See Section 9.3 of Bishop, "Pattern recognition and machine learning", 2006
/// Derived classes must implement the initialization(), eStep() and mStep() methods.
template<class T_machine, class T_sampler>
class EMTrainer : virtual public Trainer<T_machine, T_sampler>
{
public:
  virtual ~EMTrainer() {}
  

  virtual void train(T_machine& machine, const T_sampler& sampler) {
    Torch::core::info << "# EMTrainer:" << std::endl;
    
    /*
    // Check that the machine and dataset have the same feature dimensionality
    if (!checkForDimensionalityMatch()) {
      Torch::core::error << "Mismatch in dimensionality of dataset and machine" << endl;
      return false;
    }
    */
    
    initialization(machine, sampler);
    // Do the expectation-maximisation algorithm
    double average_output_previous = DBL_MIN;
    // - initial eStep (and calculate average output)
    double average_output = eStep(machine, sampler);
    
    // - iterate...
    for(int iter=1; true; ++iter) {
      
      // - save average output from last iteration
      average_output_previous = average_output;
      
      // - mStep
      mStep(machine, sampler);
      
      // - eStep (and re-calculate average output)
      average_output = eStep(machine, sampler);
      
      Torch::core::info << "# Iter " << iter << ": " 
        << average_output_previous << " -> " 
        << average_output << std::endl;
      
      // - terminate if converged 
      if (fabs((average_output_previous - average_output)/average_output_previous) <= convergence_threshold) {
        Torch::core::info << "# EM terminated: likelihood converged" << std::endl;
        break;
      }
      
       // - terminate if done max iterations 
      if (max_iterations > 0 && iter >= max_iterations-1) {
        Torch::core::info << "# EM terminated: maximum number of iterations reached." << std::endl;
        break;
      }
    }
  }

  /// This method is called before the EM algorithm 
  virtual void initialization(T_machine& machine, const T_sampler& sampler) = 0;
  
  /// Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters.
  /// Also, calculate the average output of the Machine given these parameters.
  /// @return The average output of the Machine across the dataset.
  ///         The EM algorithm will terminate once the change in average_output
  ///         is less than the convergence_threshold.
  virtual double eStep(T_machine& machine, const T_sampler& sampler) = 0;
  
  /// Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)
  virtual void mStep(T_machine& machine, const T_sampler& sampler) = 0;

  /// Set convergence threshold
  void setConvergenceThreshold(double threshold) {
    convergence_threshold = threshold;
  }

  /// Get convergence threshold
  double getConvergenceThreshold() {
    return convergence_threshold;
  }

  /// Set max iterations
  void setMaxIterations(int max_iterations) {
    this->max_iterations = max_iterations;
  }

  /// Get max iterations
  int getMaxIterations() {
    return max_iterations;
  }

protected:
  double convergence_threshold;
  int max_iterations;

  EMTrainer(double convergence_threshold = 0.001, int max_iterations = 10) :
    convergence_threshold(convergence_threshold), max_iterations(max_iterations) {
  }
};

}
}

#endif // EMTRAINER_H
