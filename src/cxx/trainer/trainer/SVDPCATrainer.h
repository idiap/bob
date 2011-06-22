/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 09 Jun 2011 12:48:43 CEST
 *
 * @brief Principal Component Analysis implemented with Singular Value
 * Decomposition (lapack)
 */

#ifndef TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H
#define TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H

#include "machine/LinearMachine.h"
#include "io/Arrayset.h"

namespace Torch { namespace trainer {
  
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
      virtual void train(Torch::machine::LinearMachine& machine, 
          const Torch::io::Arrayset& data) const;

      /**
       * Trains the LinearMachine to perform the KLT. The resulting machine
       * will have the eigen-vectors of the covariance matrix arranged by
       * decreasing energy automatically. You don't need to sort the results.
       * Also returns the eigen values of the covariance matrix so you can use
       * that to choose which components to keep.
       */
      virtual void train(Torch::machine::LinearMachine& machine,
          blitz::Array<double,1>& eigen_values,
          const Torch::io::Arrayset& data) const;

    private: //representation

      bool m_zscore_convert; ///< apply std.dev. normalization to machines?

  };

}}

#endif /* TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H */
