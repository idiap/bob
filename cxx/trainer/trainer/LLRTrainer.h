/**
 * @file cxx/trainer/trainer/LLRTrainer.h
 * @date Sat Sep 1 19:16:00 2012 +0100
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Linear Logistic Regression trainer using a conjugate gradient
 *   approach.
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

#ifndef BOB_TRAINER_LLRTRAINER_H
#define BOB_TRAINER_LLRTRAINER_H

#include "machine/LinearMachine.h"
#include "io/Arrayset.h"

namespace bob { namespace trainer {
  
  /**
    * Trains a Linear Logistic Regression model using a conjugate gradient 
    * approach. Reference:
    *   "A comparison of numerical optimizers for logistic regression", 
    *   T. Minka, Unpublished draft, 2003 (revision in 2007), 
    *   http://research.microsoft.com/en-us/um/people/minka/papers/logreg/
    */
  class LLRTrainer {

    public: //api

      /**
       * Default constructor.
       */
      LLRTrainer(const double prior=0.5, 
        const double convergence_threshold=1e-5,
        const size_t max_iterations=10000);

      /**
       * Copy construction.
       */
      LLRTrainer(const LLRTrainer& other);

      /**
       * Destructor virtualisation
       */
      virtual ~LLRTrainer();

      /**
       * Copy operator
       */
      LLRTrainer& operator=(const LLRTrainer& other);

      /**
        * Getters
        */
      double getPrior() const { return m_prior; }
      double getConvergenceThreshold() const { return m_convergence_threshold; }
      size_t getMaxIterations() const { return m_max_iterations; }

      /**
        * Setters
        */
      void setPrior(const double prior) { m_prior = prior; }
      void setConvergenceThreshold(const double convergence_threshold)
      { m_convergence_threshold = convergence_threshold; }
      void setMaxIterations(const size_t max_iterations) 
      { m_max_iterations = max_iterations; }

      /**
       * Trains the LinearMachine to perform the Linear Logistic Regression.
       */
      virtual void train(bob::machine::LinearMachine& machine, 
          const bob::io::Arrayset& data1, const bob::io::Arrayset& data2) const;

    private: //representation
      // Attributes
      double m_prior;
      double m_convergence_threshold;
      size_t m_max_iterations;
  };

}}

#endif /* BOB_TRAINER_LLRTRAINER_H */
