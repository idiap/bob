/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 31 May 15:08:46 2013 
 *
 * @brief Implements the Square Error Cost function
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

#ifndef BOB_TRAINER_SQUAREERROR_H 
#define BOB_TRAINER_SQUAREERROR_H 

#include "bob/trainer/Cost.h"

namespace bob { namespace trainer {
  /**
   * @ingroup TRAINER
   * @{
   */

  /**
   * Calculates the Square-Error between output and target. The square error is
   * defined as follows:
   *
   * \f[
   *    J = \frac{(\hat{y} - y)^2}{2}
   * \f]
   *
   * where \f$\hat{y}\f$ is the output estimated by your machine and \f$y\f$ is
   * the expected output.
   */
  class SquareError: public Cost {
    
    public:

      /**
       * Builds a SquareError functor with an existing activation function.
       */
      SquareError(boost::shared_ptr<bob::machine::Activation> actfun);

      /**
       * Virtualized destructor
       */
      virtual ~SquareError();

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

  };

  /**
   * @}
   */
}}

#endif /* BOB_TRAINER_SQUAREERROR_H */
