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
       *
       * @param batch_size The number of examples passed at each iteration.
       * This should be a big number (tens of samples) - Resilient
       * Back-propagation is a <b>batch</b> algorithm, it requires large sample
       * sizes
       *
       * @param cost This is the cost function to use for the current training.
       *
       * @note Good values for batch sizes are tens of samples. This may affect
       * the convergence.
       */
      MLPRPropTrainer(size_t batch_size,
          boost::shared_ptr<bob::trainer::Cost> cost);

      /**
       * @brief Initializes a new MLPRPropTrainer trainer according to a given
       * machine settings and a training batch size. 
       *
       * @param batch_size The number of examples passed at each iteration.
       * This should be a big number (tens of samples) - Resilient
       * Back-propagation is a <b>batch</b> algorithm, it requires large sample
       * sizes
       *
       * @param cost This is the cost function to use for the current training.
       *
       * @param machine Clone this machine weights and prepare the trainer
       * internally mirroring machine properties.
       *
       * @note Good values for batch sizes are tens of samples. This may affect
       * the convergence.
       */
      MLPRPropTrainer(size_t batch_size, 
          boost::shared_ptr<bob::trainer::Cost> cost,
          const bob::machine::MLP& machine);

      /**
       * @brief Initializes a new MLPRPropTrainer trainer according to a
       * given machine settings and a training batch size.
       *
       * @param batch_size The number of examples passed at each iteration.
       * This should be a big number (tens of samples) - Resilient
       * Back-propagation is a <b>batch</b> algorithm, it requires large sample
       * sizes
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
       * @param train_biases A boolean, indicating if we need to train the
       * biases or not.
       *
       * You can also change default values for the learning rate and momentum.
       * By default we train w/o any momenta.
       *
       * If you want to adjust a potential learning rate decay, you can and
       * should do it outside the scope of this trainer, in your own way.
       */
      MLPRPropTrainer(size_t batch_size, 
          boost::shared_ptr<bob::trainer::Cost> cost,
          const bob::machine::MLP& machine,
          bool train_biases);

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

      /**
       * @brief Initialize the internal buffers for the current machine
       */
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

      /**
       * Accessors for algorithm parameters
       */

      /**
       * @brief Gets the de-enforcement parameter (default is 0.5)
       */
      double getEtaMinus() const { return m_eta_minus; }

      /**
       * @brief Sets the de-enforcement parameter (default is 0.5)
       */
      void setEtaMinus(double v) { m_eta_minus = v;    }

      /**
       * @brief Gets the enforcement parameter (default is 1.2)
       */
      double getEtaPlus() const { return m_eta_plus; }

      /**
       * @brief Sets the enforcement parameter (default is 1.2)
       */
      void setEtaPlus(double v) { m_eta_plus = v;    }
      
      /**
       * @brief Gets the initial weight update (default is 0.1)
       */
      double getDeltaZero() const { return m_delta_zero; }
      
      /**
       * @brief Sets the initial weight update (default is 0.1)
       */
      void setDeltaZero(double v) { m_delta_zero = v;    }

      /**
       * @brief Gets the minimal weight update (default is 1e-6)
       */
      double getDeltaMin() const { return m_delta_min; }
      
      /**
       * @brief Sets the minimal weight update (default is 1e-6)
       */
      void setDeltaMin(double v) { m_delta_min = v;    }

      /**
       * @brief Gets the maximal weight update (default is 50.0)
       */
      double getDeltaMax() const { return m_delta_max; }

      /**
       * @brief Sets the maximal weight update (default is 50.0)
       */
      void setDeltaMax(double v) { m_delta_max = v;    }

      /**
       * @brief Returns the deltas
       */
      const std::vector<blitz::Array<double,2> >& getDeltas() const { return m_delta; }

      /**
       * @brief Returns the deltas
       */
      const std::vector<blitz::Array<double,1> >& getBiasDeltas() const { return m_delta_bias; }

      /**
       * @brief Sets the deltas
       */
      void setDeltas(const std::vector<blitz::Array<double,2> >& v);

      /**
       * @brief Sets the deltas for a given index
       */
      void setDelta(const blitz::Array<double,2>& v, const size_t index);

      /**
       * @brief Sets the bias deltas
       */
      void setBiasDeltas(const std::vector<blitz::Array<double,1> >& v);

      /**
       * @brief Sets the bias deltas for a given index
       */
      void setBiasDelta(const blitz::Array<double,1>& v, const size_t index);

      /**
       * @brief Returns the derivatives of the cost wrt. the weights
       */
      const std::vector<blitz::Array<double,2> >& getPreviousDerivatives() const { return m_prev_deriv; }

      /**
       * @brief Returns the derivatives of the cost wrt. the biases
       */
      const std::vector<blitz::Array<double,1> >& getPreviousBiasDerivatives() const { return m_prev_deriv_bias; }

      /**
       * @brief Sets the previous derivatives of the cost
       */
      void setPreviousDerivatives(const std::vector<blitz::Array<double,2> >& v);

      /**
       * @brief Sets the previous derivatives of the cost of a given index
       */
      void setPreviousDerivative(const blitz::Array<double,2>& v, const size_t index);

      /**
       * @brief Sets the previous derivatives of the cost (biases)
       */
      void setPreviousBiasDerivatives(const std::vector<blitz::Array<double,1> >& v);

      /**
       * @brief Sets the previous derivatives of the cost (biases) of a given
       * index
       */
      void setPreviousBiasDerivative(const blitz::Array<double,1>& v, const size_t index);

    private: //representation

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

      double m_eta_minus; ///< de-enforcement parameter (0.5)
      double m_eta_plus;  ///< enforcement parameter (1.2)
      double m_delta_zero;///< initial value for the weight change (0.1)
      double m_delta_min; ///< minimum value for the weight change (1e-6)
      double m_delta_max; ///< maximum value for the weight change (50.0)

      std::vector<blitz::Array<double,2> > m_delta; ///< R-prop weights deltas
      std::vector<blitz::Array<double,1> > m_delta_bias; ///< R-prop biases deltas

      std::vector<blitz::Array<double,2> > m_prev_deriv; ///< prev.weight deriv.
      std::vector<blitz::Array<double,1> > m_prev_deriv_bias; ///< pr.bias der.
  };

  /**
   * @}
   */
} }

#endif /* BOB_TRAINER_MLPRPROPTRAINER_H */
