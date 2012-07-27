/**
 * @file visioner/src/diag_exp_loss.cc
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

#include "visioner/model/losses/diag_exp_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  scalar_t DiagExpLoss::error(scalar_t target, scalar_t score) const
  {
    return classification_error(target, score, 0.0);
  }

  // Compute the loss value & derivatives
  void DiagExpLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value) const
  {
    const scalar_t eval = my_exp(- target * score);

    value = eval;
  }
  void DiagExpLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1) const
  {
    const scalar_t eval = my_exp(- target * score);

    value = eval;
    deriv1 = - target * eval;
  }
  void DiagExpLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1, scalar_t& deriv2) const
  {
    const scalar_t eval = my_exp(- target * score);

    value = eval;
    deriv1 = - target * eval;
    deriv2 = target * target * eval;
  }

}}
