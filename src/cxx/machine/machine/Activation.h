/**
 * @file cxx/machine/machine/Activation.h
 * @date Thu Jul 7 16:49:35 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Activation functions for linear and MLP machines.
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_MACHINE_ACTIVATION_H 
#define BOB_MACHINE_ACTIVATION_H


namespace bob { namespace machine {

  typedef enum Activation {
    LINEAR = 0, //Linear: y = x [this is the default]
    TANH = 1, //Hyperbolic tangent: y = (e^x - e^(-x))/(e^x + e^(-x))
    LOG = 2 //Logistic function: y = 1/(1 + e^(-x))
  } Activation;

  inline double linear(double x) { return x; }
  inline double logistic(double x) { return 1.0 / (1.0 + std::exp(-x)); }
  //tanh already exists in the standard cmath

  /**
   * The next functions are the derivative versions of all activation
   * functions, expressed in terms of the same 'x' as the input value used for
   * the activation. Here are the examples:
   *
   *  F           | derivative as a function of F
   *  ------------+------------------------------------------
   *  tanh(x)     | tanh'(x) = 1-(tanh(x))^2 = 1-F^2
   *  logistic(x) | logistic'(x) = logistic(x) * (1-logistic(x)) = F*(1-F)
   *  linear(x)   | linear'(x) = 1
   */
  inline double linear_derivative(double x) { return 1; }
  inline double tanh_derivative(double x) { return 1-(x*x); }
  inline double logistic_derivative(double x) { return x*(1-x); }

}}
      
#endif /* BOB_MACHINE_ACTIVATION_H */

