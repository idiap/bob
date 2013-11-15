/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 31 May 23:52:08 2013 CEST 
 *
 * @brief Implementation of the cross entropy loss function
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/trainer/CrossEntropyLoss.h"

namespace bob { namespace trainer {

  CrossEntropyLoss::CrossEntropyLoss(boost::shared_ptr<bob::machine::Activation> actfun)
    : m_actfun(actfun),
      m_logistic_activation(m_actfun->unique_identifier() == "bob.machine.Activation.Logistic") {}

  CrossEntropyLoss::~CrossEntropyLoss() {}

  double CrossEntropyLoss::f (double output, double target) const {
    return - (target * std::log(output)) - ((1-target)*std::log(1-output));
  }

  double CrossEntropyLoss::f_prime (double output, double target) const {
    return (output-target) / (output * (1-output));
  }

  double CrossEntropyLoss::error (double output, double target) const {
    return m_logistic_activation? (output - target) : m_actfun->f_prime_from_f(output) * f_prime(output, target);
  }

  std::string CrossEntropyLoss::str() const {
    std::string retval = "J = - target*log(output) - (1-target)*log(1-output) (cross-entropy loss)";
    if (m_logistic_activation) retval += " [+ logistic activation]";
    else retval += " [+ unknown activation]";
    return retval;
  }

}}
