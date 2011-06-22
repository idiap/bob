/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jun 09:33:28 2011 CEST
 *
 * @brief Implements a multi-class Fisher/LDA linear machine Training using
 * Singular Value Decomposition (SVD). For more information on Linear Machines
 * and associated methods, please consult Bishop, Machine Learning and Pattern
 * Recognition chapter 4.
 */

#ifndef TORCH5SPRO_TRAINER_FISHER_LDA_TRAINER_H
#define TORCH5SPRO_TRAINER_FISHER_LDA_TRAINER_H

#include <vector>

#include "trainer/Trainer.h"
#include "machine/LinearMachine.h"
#include "io/Arrayset.h"

namespace Torch { namespace trainer {
  
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
      virtual void train(Torch::machine::LinearMachine& machine, 
          const std::vector<Torch::io::Arrayset>& data) const;

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
      virtual void train(Torch::machine::LinearMachine& machine,
          blitz::Array<double,1>& eigen_values,
          const std::vector<Torch::io::Arrayset>& data) const;

  };

} }

#endif /* TORCH5SPRO_TRAINER_FISHER_LDA_TRAINER_H */
