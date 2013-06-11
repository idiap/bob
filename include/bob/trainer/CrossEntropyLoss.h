/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 31 May 15:08:46 2013 
 *
 * @brief Implements the Cross Entropy Loss function
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

#ifndef BOB_TRAINER_CROSSENTROPYLOSS_H 
#define BOB_TRAINER_CROSSENTROPYLOSS_H 

#include "bob/trainer/Cost.h"

namespace bob { namespace trainer {
  /**
   * @ingroup TRAINER
   * @{
   */

  /**
   * Calculates the Cross-Entropy Loss between output and target. The cross
   * entropy loss is defined as follows:
   *
   * \f[
   *    J = - y \cdot \log{(\hat{y})} - (1-y) \log{(1-\hat{y})}
   * \f]
   *
   * where \f$\hat{y}\f$ is the output estimated by your machine and \f$y\f$ is
   * the expected output.
   */
  class CrossEntropyLoss: public Cost {
    
    public:

      /**
       * Constructor
       *
       * @param actfun Sets the underlying activation function used for error
       * calculation. A special case is foreseen for using this loss function
       * with a logistic activation. In this case, a mathematical
       * simplification is possible in which error() can benefit increasing the
       * numerical stability of the training process. The simplification goes
       * as follows:
       *
       * \f[
       *    b = \delta \cdot \varphi'(z)
       * \f]
       *
       * But, for the CrossEntropyLoss: 
       *
       * \f[
       *    \delta = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y}}
       * \f]
       *
       * and \f$\varphi'(z) = \hat{y} - (1 - \hat{y})\f$, so:
       *
       * \f[
       *    b = \hat{y} - y
       * \f]
       */
      CrossEntropyLoss(boost::shared_ptr<bob::machine::Activation> actfun);

      /**
       * Virtualized destructor
       */
      virtual ~CrossEntropyLoss();

      /**
       * Tells if this CrossEntropyLoss is set to operate together with a
       * bob::machine::LogisticActivation.
       */
      bool logistic_activation() const { return m_logistic_activation; }

      /**
       * Computes cost, given the current output of the linear machine or MLP
       * and the expected output.
       *
       * @param output Real output from the linear machine or MLP
       *
       * @param target Target output you are training to achieve
       *
       * @return The cost
       */
      virtual double f (double output, double target) const;

      /**
       * Computes the derivative of the cost w.r.t. output.
       * 
       * @param output Real output from the linear machine or MLP
       *
       * @param target Target output you are training to achieve
       *
       * @return The calculated error
       */
      virtual double f_prime (double output, double target) const;

      /**
       * Computes the back-propagated errors for a given MLP <b>output</b>
       * layer, given its activation function and activation values - i.e., the
       * error back-propagated through the last layer neurons up to the
       * synapses connecting the last hidden layer to the output layer.
       *
       * This entry point allows for optimization in the calculation of the
       * back-propagated errors in cases where there is a possibility of
       * mathematical simplification when using a certain combination of
       * cost-function and activation. For example, using a ML-cost and a
       * logistic activation function.
       *
       * @param output Real output from the linear machine or MLP
       * @param target Target output you are training to achieve
       *
       * @return The calculated error, backpropagated to before the output
       * neuron.
       */
      virtual double error (double output, double target) const;

      /**
       * Returns a stringified representation for this Cost function
       */
      virtual std::string str() const;

    private: //representation

      boost::shared_ptr<bob::machine::Activation> m_actfun; //act. function
      bool m_logistic_activation; ///< if 'true', simplify backprop_error()

  };

  /**
   * @}
   */
}}

#endif /* BOB_TRAINER_CROSSENTROPYLOSS_H */
