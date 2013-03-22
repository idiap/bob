/**
 * @file bob/trainer/SVDPCATrainer.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Principal Component Analysis implemented with Singular Value
 * Decomposition (lapack)
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

#ifndef BOB_TRAINER_SVDPCA_TRAINER_H
#define BOB_TRAINER_SVDPCA_TRAINER_H

#include <bob/machine/LinearMachine.h>

namespace bob { namespace trainer {
  /**
   * @ingroup TRAINER
   * @{
   */
  
  /**
   * @brief Sets a linear machine to perform the Karhunen-Loève Transform 
   * (KLT) on a given dataset using Singular Value Decomposition (SVD). 
   *
   * References:
   * 1. Eigenfaces for Recognition, Turk & Pentland, Journal of Cognitive
   *    Neuroscience (1991) Volume: 3, Issue: 1, Publisher: MIT Press, 
   *    Pages: 71-86
   * 2. http://en.wikipedia.org/wiki/Singular_value_decomposition
   * 3. http://en.wikipedia.org/wiki/Principal_component_analysis
   */
  class SVDPCATrainer {

    public: //api

      /**
       * Initializes a new SVD/PCD trainer. The training stage will place the
       * resulting principal components in the linear machine and set it up to
       * extract the variable means automatically. As an option, you may preset
       * the trainer so that the normalization performed by the resulting
       * linear machine also divides the variables by the standard deviation of
       * each variable ensemble.
       */
      SVDPCATrainer();

      /**
       * Copy construction.
       */
      SVDPCATrainer(const SVDPCATrainer& other);

      /**
       * Destructor virtualisation
       */
      virtual ~SVDPCATrainer();

      /**
       * Copy operator
       */
      SVDPCATrainer& operator=(const SVDPCATrainer& other);

      /**
       * Trains the LinearMachine to perform the KLT. The resulting machine
       * will have the eigen-vectors of the covariance matrix arranged by
       * decreasing energy automatically. You don't need to sort the results.
       */
      virtual void train(bob::machine::LinearMachine& machine, 
          const blitz::Array<double,2>& data) const;

      /**
       * Trains the LinearMachine to perform the KLT. The resulting machine
       * will have the eigen-vectors of the covariance matrix arranged by
       * decreasing energy automatically. You don't need to sort the results.
       * Also returns the eigen values of the covariance matrix so you can use
       * that to choose which components to keep.
       */
      virtual void train(bob::machine::LinearMachine& machine,
          blitz::Array<double,1>& eigen_values,
          const blitz::Array<double,2>& data) const;

    private: //representation

  };

  /**
   * @}
   */
}}

#endif /* BOB_TRAINER_SVDPCA_TRAINER_H */
