/**
 * @file cxx/trainer/trainer/MLPBackPropTrainer.h
 * @date Mon Jul 18 18:11:22 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A MLP trainer based on vanilla back-propagation. You can get an
 * overview of this method at "Pattern Recognition and Machine Learning"
 * by C.M. Bishop (Chapter 5).
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

#ifndef BOB_TRAINER_MLPBACKPROPTRAINER_H 
#define BOB_TRAINER_MLPBACKPROPTRAINER_H

#include <vector>
#include <boost/function.hpp>

#include "bob/io/Arrayset.h"
#include "bob/machine/MLP.h"

namespace bob { namespace trainer {

  /**
   * Sets an MLP to perform discrimination based on vanilla error
   * back-propagation as defined in "Pattern Recognition and Machine Learning"
   * by C.M. Bishop, chapter 5.
   */
  class MLPBackPropTrainer {

    public: //api

      /**
       * Initializes a new MLPBackPropTrainer trainer according to a given
       * machine settings and a training batch size.
       *
       * Good values for batch sizes are tens of samples. BackProp is not
       * necessarily a "batch" training algorithm, but performs in a smoother
       * if the batch size is larger. This may also affect the convergence.
       *
       * You can also change default values for the learning rate and momentum.
       * By default we train w/o any momenta.
       *
       * If you want to adjust a potential learning rate decay, you can and
       * should do it outside the scope of this trainer, in your own way.
       *
       * Here is an overview of the backprop algorithm executed by this
       * trainer:
       *
       * -# Take the <em>local gradient</em> of a neuron
       *    @f[ l^{(l)} @f]
       * -# Multiply that value by the <em>output</em> of the previous layer;
       *    @f[
       *    l^{(l)} \times y^{(l-1)}
       *    @f]
       * -# Multiply the result of the previous step by the learning rate;
       *    @f[
       *    \eta \times l^{(l)} \times y^{(l-1)}
       *    @f]
       * -# Add the result of the previous setup to the current weight,
       *    possibly weighting the sum with a momentum ponderator.
       *    @f[
       *    w_{n+1} = (1-\mu) \times (w_{n} + \eta \times l^{(l)} 
       *    \times y^{(l-1)}) + (\mu) \times w_{n-1}
       *    @f]
       */
      MLPBackPropTrainer(const bob::machine::MLP& machine, size_t batch_size);

      /**
       * Destructor virtualisation
       */
      virtual ~MLPBackPropTrainer();
      
      /**
       * Copy construction.
       */
      MLPBackPropTrainer(const MLPBackPropTrainer& other);

      /**
       * Copy operator
       */
      MLPBackPropTrainer& operator=(const MLPBackPropTrainer& other);

      /**
       * Re-initializes the whole training apparatus to start training a new
       * machine. This will effectively reset all Delta matrices to their
       * intial values and set the previous derivatives to zero.
       */
      void reset();

      /**
       * Gets the batch size
       */
      size_t getBatchSize() const { return m_target.extent(0); }

      /**
       * Sets the batch size
       */
      void setBatchSize(size_t batch_size);

      /**
       * Gets the current learning rate
       */
      double getLearningRate() const { return m_learning_rate; }

      /**
       * Sets the current learning rate
       */
      void setLearningRate(double v) { m_learning_rate = v; }

      /**
       * Gets the current momentum
       */
      double getMomentum() const { return m_momentum; }

      /**
       * Sets the current momentum
       */
      void setMomentum(double v) { m_momentum = v; }

      /**
       * Gets the current settings for bias training (defaults to true)
       */
      inline bool getTrainBiases() const { return m_train_bias; }

      /**
       * Sets the bias training option
       */
      inline void setTrainBiases(bool v) { m_train_bias = v; }

      /**
       * Checks if a given machine is compatible with my inner settings.
       */
      bool isCompatible(const bob::machine::MLP& machine) const;

      /**
       * Trains the MLP to perform discrimination. The training is executed
       * outside the machine context, but uses all the current machine layout.
       * The given machine is updated with new weights and biases on the end of
       * the training that is performed a single time. Iterate as much as you
       * want to refine the training.
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
       * This is a version of the train() method above, which does no
       * compatibility check on the input machine.
       */
      void train_(bob::machine::MLP& machine, 
          const blitz::Array<double,2>& input,
          const blitz::Array<double,2>& target);

    private: //useful methods

      /**
       * Forward step -- this is a second implementation of that used on the
       * MLP itself to allow access to some internal buffers. In our current
       * setup, we keep the "m_output"'s of every individual layer separately
       * as we are going to need them for the weight update.
       *
       * Another factor is the normalization normally applied at MLPs. We
       * ignore that here as the DataShuffler should be capable of handling
       * this in a more efficient way. You should make sure that the final MLP
       * does have the standard normalization settings applied if it was set to
       * automatically apply the standard normalization before giving me the
       * data.
       */
      void forward_step();

      /**
       * Backward step -- back-propagates the calculated error up to each
       * neuron on the first layer. This is explained on Bishop's formula 5.55
       * and 5.56, at page 244 (see also figure 5.7 for a graphical
       * representation).
       */
      void backward_step();

      /**
       * Weight update -- calculates the weight-update using derivatives as
       * required by back-prop.
       */
      void backprop_weight_update();

    private: //representation

      /// training parameters:
      double m_learning_rate;
      double m_momentum; 

      bool m_train_bias; ///< shall we be training biases? (default: true)
      size_t m_H; ///< number of hidden layers on the target machine

      std::vector<blitz::Array<double,2> > m_weight_ref; ///< machine weights
      std::vector<blitz::Array<double,1> > m_bias_ref; ///< machine biases
      
      std::vector<blitz::Array<double,2> > m_delta; ///< weight deltas
      std::vector<blitz::Array<double,1> > m_delta_bias; ///< bias deltas

      std::vector<blitz::Array<double,2> > m_prev_delta; ///< prev.weight deltas
      std::vector<blitz::Array<double,1> > m_prev_delta_bias; ///< prev. bias ds

      bob::machine::MLP::actfun_t m_actfun; ///< activation function
      bob::machine::MLP::actfun_t m_bwdfun; ///< (back) activation function
  
      /// buffers that are dependent on the batch_size
      blitz::Array<double,2> m_target; ///< target vectors
      std::vector<blitz::Array<double,2> > m_error; ///< error (+deltas)
      std::vector<blitz::Array<double,2> > m_output; ///< layer output
  };

} }

#endif /* BOB_TRAINER_MLPBACKPROPTRAINER_H */
