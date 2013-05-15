/**
 * @file bob/trainer/MLPRPropTrainer.h
 * @date Wed Jul 6 17:32:35 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey<Laurent.El-Shafey@idiap.ch>
 *
 * @brief A MLP trainer based on resilient back-propagation: A Direct Adaptive
 * Method for Faster Backpropagation Learning: The RPROP Algorithm, by Martin
 * Riedmiller and Heinrich Braun on IEEE International Conference on Neural
 * Networks, pp. 586--591, 1993.
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

#ifndef BOB_TRAINER_MLPRPROPTRAINER_H 
#define BOB_TRAINER_MLPRPROPTRAINER_H

#include <vector>
#include <boost/function.hpp>

#include <bob/machine/MLP.h>

#include "MLPBaseTrainer.h"

namespace bob { namespace trainer {
  /**
   * @ingroup TRAINER
   * @{
   */

  /**
   * @brief Sets an MLP to perform discrimination based on RProp: A Direct 
   * Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm,
   * by Martin Riedmiller and Heinrich Braun on IEEE International Conference 
   * on Neural Networks, pp. 586--591, 1993.
   */
  class MLPRPropTrainer: public MLPBaseTrainer {

    public: //api

      /**
       * @brief Initializes a new MLPRPropTrainer trainer according to a given
       * training batch size. 
       */
      MLPRPropTrainer(size_t batch_size);


      /**
       * @brief Initializes a new MLPRPropTrainer trainer according to a given
       * machine settings and a training batch size. 
       *
       * Good values for batch sizes are tens of samples. RProp is a "batch"
       * training algorithm. Do not try to set batch_size to a too-low value.
       */
      MLPRPropTrainer(const bob::machine::MLP& machine, size_t batch_size);

      /**
       * @brief Destructor virtualisation
       */
      virtual ~MLPRPropTrainer();
      
      /**
       * @brief Copy construction.
       */
      MLPRPropTrainer(const MLPRPropTrainer& other);

      /**
       * @brief Copy operator
       */
      MLPRPropTrainer& operator=(const MLPRPropTrainer& other);

      /**
       * @brief Re-initializes the whole training apparatus to start training
       * a new machine. This will effectively reset all Delta matrices to their
       * intial values and set the previous derivatives to zero as described on
       * the section II.C of the RProp paper.
       */
      void reset();

      virtual void initialize(const bob::machine::MLP& machine);

      /**
       * @brief Trains the MLP to perform discrimination. The training is
       * executed outside the machine context, but uses all the current machine
       * layout. The given machine is updated with new weights and biases on
       * the end of the training that is performed a single time. Iterate as 
       * much as you want to refine the training.
       *
       * The machine given as input is checked for compatibility with the
       * current initialized settings. If the two are not compatible, an
       * exception is thrown.
       *
       * Note: In RProp, training is done in batches. The number of rows in the
       * input (and target) determines the batch size. If the batch size
       * currently set is incompatible with the given data an exception is
       * raised.
       *       
       * Note2: The machine is not initialized randomly at each train() call.
       * It is your task to call MLP::randomize() once on the machine you
       * want to train and then call train() as many times as you think are
       * necessary. This design allows for a training criteria to be encoded
       * outside the scope of this trainer and to this type to focus only on
       input, target applying the training when requested to.
       */
      void train(bob::machine::MLP& machine, 
          const blitz::Array<double,2>& input,
          const blitz::Array<double,2>& target);

      /**
       * @brief This is a version of the train() method above, which does no
       * compatibility check on the input machine.
       */
      void train_(bob::machine::MLP& machine, 
          const blitz::Array<double,2>& input,
          const blitz::Array<double,2>& target);

    private: //useful methods

      /**
       * Weight update -- calculates the weight-update using derivatives as
       * explained in Bishop's formula 5.53, page 243.
       *
       * Note: For RProp, specifically, we only care about the derivative's
       * sign, current and the previous. This is the place where standard
       * backprop and rprop diverge.
       *
       * For extra insight, double-check the Technical Report entitled "Rprop -
       * Description and Implementation Details" by Martin Riedmiller, 1994.
       * Just browse the internet for it. Keep it under your pillow ;-)
       */
      void rprop_weight_update(bob::machine::MLP& machine,
        const blitz::Array<double,2>& input);

    private: //representation

      std::vector<blitz::Array<double,2> > m_deriv; ///< weight derivatives
      std::vector<blitz::Array<double,1> > m_deriv_bias; ///< bias derivatives

      std::vector<blitz::Array<double,2> > m_prev_deriv; ///< prev.weight deriv.
      std::vector<blitz::Array<double,1> > m_prev_deriv_bias; ///< pr.bias der.
  };

  /**
   * @}
   */
} }

#endif /* BOB_TRAINER_MLPRPROPTRAINER_H */
