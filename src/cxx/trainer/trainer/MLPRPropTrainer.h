/**
 * @author Andre Anjos <andre.anjos@idiap.ch> 
 * @date Tue 05 Jul 2011 12:04:18 CEST 
 *
 * @brief A MLP trainer based on resilient back-propagation: A Direct Adaptive
 * Method for Faster Backpropagation Learning: The RPROP Algorithm, by Martin
 * Riedmiller and Heinrich Braun on IEEE International Conference on Neural
 * Networks, pp. 586--591, 1993.
 */

#ifndef TORCH_TRAINER_MLPRPROPTRAINER_H 
#define TORCH_TRAINER_MLPRPROPTRAINER_H

#include <vector>
#include <boost/function.hpp>

#include "io/Arrayset.h"
#include "machine/MLP.h"
#include "trainer/MLPStopCriteria.h"

namespace Torch { namespace trainer {

  /**
   * Sets an MLP to perform discrimination based on RProp: A Direct Adaptive
   * Method for Faster Backpropagation Learning: The RPROP Algorithm, by Martin
   * Riedmiller and Heinrich Braun on IEEE International Conference on Neural
   * Networks, pp. 586--591, 1993.
   */
  class MLPRPropTrainer {

    public: //api

      /**
       * Initializes a new MLPRPropTrainer trainer, passing a stop criteria to
       * stop the training.
       */
      MLPRPropTrainer(const MLPStopCriteria& s);

      /**
       * If you decide to do so, you can create a trainer which will train for
       * a fixed number of iterations. This has the same effect as creating a
       * trainer with a NumberOfIterationsCriteria() object.
       */
      MLPRPropTrainer(size_t max_iterations);

      /**
       * Destructor virtualisation
       */
      virtual ~MLPRPropTrainer();
      
      /**
       * Copy construction.
       */
      MLPRPropTrainer(const MLPRPropTrainer& other);

      /**
       * Copy operator
       */
      MLPRPropTrainer& operator=(const MLPRPropTrainer& other);

      /**
       * Trains the MLP to perform discrimination.
       *
       * Each input arrayset represents data from a given input class. One
       * should also give a matching target for each of the input classes.
       * Training will stop depending on your selected criteria set during
       * object instantiation.
       *
       * Note that for RProp, training is done in batches. You should set the
       * batch size.
       */
      virtual void train(Torch::machine::MLP& machine,
          const std::vector<Torch::io::Arrayset>& train_data,
          const std::vector<Torch::io::Array>& train_target,
          size_t batch_size) const;

    private: //representation

      MLPStopCriteria m_stop; ///< stopping criteria for MLP training

  };

} }

#endif /* TORCH_TRAINER_MLPRPROPTRAINER_H */
