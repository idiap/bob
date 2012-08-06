/**
 * @file visioner/src/diag_symlog_loss.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
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

#include "visioner/model/losses/diag_symlog_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  double DiagSymLogLoss::error(double target, double score) const
  {
    return regression_error(target, score, 0.0);
  }

  // Compute the loss value & derivatives
  void DiagSymLogLoss::eval(
      double target, double score,
      double& value) const
  {
    static const double delta = my_log(4.0);

    const double eval = my_exp(score - target);

    value = my_log(2.0 + eval + 1.0 / eval) - delta;
  }
  void DiagSymLogLoss::eval(
      double target, double score,
      double& value, double& deriv1) const
  {
    static const double delta = my_log(4.0);

    const double eval = my_exp(score - target);
    const double norm = 1.0 / (1.0 + eval);

    value = my_log(2.0 + eval + 1.0 / eval) - delta;
    deriv1 = (eval - 1) * norm;
  }
  void DiagSymLogLoss::eval(
      double target, double score,
      double& value, double& deriv1, double& deriv2) const
  {
    static const double delta = my_log(4.0);

    const double eval = my_exp(score - target);
    const double norm = 1.0 / (1.0 + eval);

    value = my_log(2.0 + eval + 1.0 / eval) - delta;
    deriv1 = (eval - 1) * norm;
    deriv2 = 2.0 * eval * norm * norm;
  }

}}
