/**
 * @file bob/trainer/FisherLDATrainer.h
 * @date Sat Jun 4 21:38:59 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements a multi-class Fisher/LDA linear machine Training using
 * Singular Value Decomposition (SVD). For more information on Linear Machines
 * and associated methods, please consult Bishop, Machine Learning and Pattern
 * Recognition chapter 4.
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

#ifndef BOB_TRAINER_FISHER_LDA_TRAINER_H
#define BOB_TRAINER_FISHER_LDA_TRAINER_H

#include <vector>

#include <bob/trainer/Trainer.h>
#include <bob/machine/LinearMachine.h>

namespace bob { namespace trainer { 
  /**
   * @ingroup TRAINER
   * @{
   */

  /**
   * Sets a linear machine to perform the Fisher/LDA decomposition. References:
   *
   * 1. Bishop, Machine Learning and Pattern Recognition chapter 4.
   * 2. http://en.wikipedia.org/wiki/Linear_discriminant_analysis
   */
  class FisherLDATrainer {

    public: //api

      /**
       * Initializes a new Fisher/LDA trainer. The training stage will place
       * the resulting fisher components in the linear machine and set it up
       * to extract the variable means automatically.
       */
      FisherLDATrainer();

      /**
       * Destructor virtualisation
       */
      virtual ~FisherLDATrainer();
      
      /**
       * Copy construction.
       */
      FisherLDATrainer(const FisherLDATrainer& other);

      /**
       * Copy operator
       */
      FisherLDATrainer& operator=(const FisherLDATrainer& other);

      /**
       * Trains the LinearMachine to perform Fisher/LDA discrimination. The
       * resulting machine will have the eigen-vectors of the Sigma-1 * Sigma_b
       * product, arranged by decreasing energy.
       *
       * Each input arrayset represents data from a given input class.
       *
       * Note we set only the N-1 eigen vectors in the linear machine since the
       * last eigen value should be zero anyway. You can compress the machine
       * output further using resize() if necessary.
       */
      virtual void train(bob::machine::LinearMachine& machine, 
          const std::vector<blitz::Array<double,2> >& data) const;

      /**
       * Trains the LinearMachine to perform Fisher/LDA discrimination. The
       * resulting machine will have the eigen-vectors of the Sigma-1 * Sigma_b
       * product, arranged by decreasing energy. You don't need to sort the
       * results. Also returns the eigen values of the covariance matrix
       * product so you can use that to choose which components to keep.
       *
       * Each input arrayset represents data from a given input class.
       *
       * Note we set only the N-1 eigen vectors in the linear machine since the
       * last eigen value should be zero anyway. You can compress the machine
       * output further using resize() if necessary.
       */
      virtual void train(bob::machine::LinearMachine& machine,
          blitz::Array<double,1>& eigen_values,
          const std::vector<blitz::Array<double,2> >& data) const;

  };

  /**
   * @}
   */
} }

#endif /* BOB_TRAINER_FISHER_LDA_TRAINER_H */
