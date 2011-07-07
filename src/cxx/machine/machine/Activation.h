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

}}
      
#endif /* TORCH_MACHINE_ACTIVATION_H */

