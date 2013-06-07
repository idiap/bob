/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 31 May 23:52:08 2013 CEST 
 *
 * @brief Implementation of the cross entropy loss function
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

#include "bob/trainer/CrossEntropyLoss.h"

namespace bob { namespace trainer {

  CrossEntropyLoss::CrossEntropyLoss(bool logistic_activation)
    : m_logistic_activation(logistic_activation) {}

  CrossEntropyLoss::~CrossEntropyLoss() {}

  double CrossEntropyLoss::f (double output, double target) const {
    return - (target * std::log(output)) - ((1-target)*std::log(1-output));
  }

  double CrossEntropyLoss::f_prime (double output, double target) const {
    return (output-target) / (output * (1-output));
  }

  double CrossEntropyLoss::error (double output, double target,
      const boost::shared_ptr<bob::machine::Activation>& actfun) const {
    return m_logistic_activation? (output - target) : actfun->f_prime_from_f(output) * f_prime(output, target);
  }

  std::string CrossEntropyLoss::str() const {
    std::string retval = "J = - target*log(output) - (1-target)*log(1-output) (cross-entropy loss)";
    if (m_logistic_activation) retval += " [+ logistic activation]";
    else retval += " [+ unknown activation]";
    return retval;
  }

}}
