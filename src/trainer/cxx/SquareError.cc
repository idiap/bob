/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 31 May 18:07:53 2013
 *
 * @brief Implementation of the squared error cost function
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

#include <cmath>

#include "bob/trainer/SquareError.h"

namespace bob { namespace trainer {

  SquareError::SquareError() {}

  SquareError::~SquareError() {}

  double SquareError::f (double output, double target) const {
    return 0.5 * std::pow(output-target, 2);
  }

  double SquareError::f_prime (double output, double target) const {
    return output - target;
  }

  double SquareError::error (double output, double target,
      const boost::shared_ptr<bob::machine::Activation>& actfun) const {
    return actfun->f_prime(output) * f_prime(output, target);
  }

  std::string SquareError::str() const {
    return "J = (output-target)^2 / 2 (square error)";
  }

}}
