/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  7 Jul 08:58:41 2011 
 *
 * @brief Activation functions for linear and MLP machines.
 */

#ifndef TORCH_MACHINE_ACTIVATION_H 
#define TORCH_MACHINE_ACTIVATION_H


namespace Torch { namespace machine {

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
      
#endif /* TORCH_MACHINE_ACTIVATION_H */

