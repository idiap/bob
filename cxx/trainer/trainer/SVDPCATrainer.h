/**
 * @file cxx/trainer/trainer/SVDPCATrainer.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Principal Component Analysis implemented with Singular Value
 * Decomposition (lapack)
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

#ifndef BOB5SPRO_TRAINER_SVDPCA_TRAINER_H
#define BOB5SPRO_TRAINER_SVDPCA_TRAINER_H

#include "machine/LinearMachine.h"
#include "io/Arrayset.h"

namespace bob { namespace trainer {
  
  /**
   * Sets a linear machine to perform the Karhunen-Loève Transform (KLT) on a
   * given dataset using Singular Value Decomposition (SVD). References:
   *
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
       *
       * @param zscore_convert If set to 'true' set up the resulting linear
       * machines to also perform zscore convertion. This will make the input
       * data to be divided by the train data standard deviation after mean
       * subtraction.
       */
      SVDPCATrainer(bool zscore_convert);

      /**
       * Default constructor. This is equivalent to calling 
       * SVDPCATrainer(false).
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
          const bob::io::Arrayset& data) const;

      /**
       * Trains the LinearMachine to perform the KLT. The resulting machine
       * will have the eigen-vectors of the covariance matrix arranged by
       * decreasing energy automatically. You don't need to sort the results.
       * Also returns the eigen values of the covariance matrix so you can use
       * that to choose which components to keep.
       */
      virtual void train(bob::machine::LinearMachine& machine,
          blitz::Array<double,1>& eigen_values,
          const bob::io::Arrayset& data) const;

    private: //representation

      bool m_zscore_convert; ///< apply std.dev. normalization to machines?

  };

}}

#endif /* BOB5SPRO_TRAINER_SVDPCA_TRAINER_H */
