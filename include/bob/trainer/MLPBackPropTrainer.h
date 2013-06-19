/**
 * @file bob/trainer/MLPBackPropTrainer.h
 * @date Mon Jul 18 18:11:22 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief A MLP trainer based on vanilla back-propagation. You can get an
 * overview of this method at "Pattern Recognition and Machine Learning"
 * by C.M. Bishop (Chapter 5).
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

#ifndef BOB_TRAINER_MLPBACKPROPTRAINER_H 
#define BOB_TRAINER_MLPBACKPROPTRAINER_H

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
   * @brief Sets an MLP to perform discrimination based on vanilla error
   * back-propagation as defined in "Pattern Recognition and Machine Learning"
   * by C.M. Bishop, chapter 5.
   */
  class MLPBackPropTrainer: public MLPBaseTrainer {

    public: //api

      /**
       * @brief Initializes a new MLPBackPropTrainer trainer according to a
       * given training batch size.
       *
       * @param batch_size The number of examples passed at each iteration. If
       * you set this to 1, then you are implementing stochastic training.
       *
       * @param cost This is the cost function to use for the current training.
       *
       * @note Good values for batch sizes are tens of samples. This may affect
       * the convergence.
       */
      MLPBackPropTrainer(size_t batch_size,
          boost::shared_ptr<bob::trainer::Cost> cost);

      /**
       * @brief Initializes a new MLPBackPropTrainer trainer according to a
       * given machine settings and a training batch size.
       *
       * @param batch_size The number of examples passed at each iteration. If
       * you set this to 1, then you are implementing stochastic training.
       *
       * @param cost This is the cost function to use for the current training.
       *
       * @param machine Clone this machine weights and prepare the trainer
       * internally mirroring machine properties.
       *
       * @note Good values for batch sizes are tens of samples. BackProp is not
       * necessarily a "batch" training algorithm, but performs in a smoother
       * if the batch size is larger. This may also affect the convergence.
       *
       * You can also change default values for the learning rate and momentum.
       * By default we train w/o any momenta.
       *
       * If you want to adjust a potential learning rate decay, you can and
       * should do it outside the scope of this trainer, in your own way.
       */
      MLPBackPropTrainer(size_t batch_size, 
          boost::shared_ptr<bob::trainer::Cost> cost,
          const bob::machine::MLP& machine);

      /**
       * @brief Destructor virtualisation
       */
      virtual ~MLPBackPropTrainer();
      
      /**
       * @brief Copy construction.
       */
      MLPBackPropTrainer(const MLPBackPropTrainer& other);

      /**
       * @brief Copy operator
       */
      MLPBackPropTrainer& operator=(const MLPBackPropTrainer& other);

      /**
       * @brief Re-initializes the whole training apparatus to start training a
       * new machine. This will effectively reset all Delta matrices to their
       * intial values and set the previous derivatives to zero.
       */
      void reset();

      /**
       * @brief Gets the current learning rate
       */
      double getLearningRate() const { return m_learning_rate; }

      /**
       * @brief Sets the current learning rate
       */
      void setLearningRate(double v) { m_learning_rate = v; }

      /**
       * @brief Gets the current momentum
       */
      double getMomentum() const { return m_momentum; }

      /**
       * @brief Sets the current momentum
       */
      void setMomentum(double v) { m_momentum = v; }

      /**
       * @brief Initialize the internal buffers for the current machine
       */
      virtual void initialize(const bob::machine::MLP& machine);

      /**
       * @brief Trains the MLP to perform discrimination. The training is
       * executed outside the machine context, but uses all the current 
       * machine layout. The given machine is updated with new weights and 
       * biases on the end of the training that is performed a single time.
       * Iterate as much as you want to refine the training.
       *
       * The machine given as input is checked for compatibility with the
       * current initialized settings. If the two are not compatible, an
       * exception is thrown.
       *
       * Note: In BackProp, training may be done in batches. The number of rows
       * in the input (and target) determines the batch size. If the batch size
       * currently set is incompatible with the given data an exception is
       * raised.
       *       
       * Note2: The machine is not initialized randomly at each train() call.
       * It is your task to call MLP::randomize() once on the machine you want
       * to train and then call train() as many times as you think are
       * necessary. This design allows for a training criteria to be encoded
       * outside the scope of this trainer and to this type to focus only on
       * input, target applying the training when requested to.
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

    protected:
      /**
       * Weight update -- calculates the weight-update using derivatives as
       * required by back-prop.
       */
      void backprop_weight_update(bob::machine::MLP& machine,
        const blitz::Array<double,2>& input);

      /// training parameters:
      double m_learning_rate;
      double m_momentum; 

      std::vector<blitz::Array<double,2> > m_prev_deriv; ///< prev.weight derivs
      std::vector<blitz::Array<double,1> > m_prev_deriv_bias; ///< prev. bias derivs
  };

  /**
   * @}
   */
} }

#endif /* BOB_TRAINER_MLPBACKPROPTRAINER_H */
