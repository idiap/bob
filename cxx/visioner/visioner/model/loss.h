/**
 * @file visioner/visioner/model/loss.h
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

#ifndef BOB_VISIONER_LOSS_H
#define BOB_VISIONER_LOSS_H

#include "visioner/model/param.h"

namespace bob { namespace visioner {

  /**
   * Generic multivariate loss function of two parameters: the target value to
   * predict and the current score estimation.
   *
   * NB: The 'grad' and 'hess' objects should be a-priori resized to <size, 1>
   * and <size, size> respectively, before calling <Loss::eval>.
   */
  class Loss : public Parametrizable {

    public:

      // Constructor
      Loss(const param_t& param = param_t())
        : Parametrizable(param) { }

      // Destructor
      virtual ~Loss() {}

      // Clone the object
      virtual boost::shared_ptr<Loss> clone() const = 0;

      // Compute the error (associated to the loss)
      virtual double error(
          const double* targets, const double* scores, uint64_t size) const = 0;

      // Compute the loss value & derivatives
      virtual void eval(const double* targets, const double* scores, uint64_t size, double& value) const = 0;

      virtual void eval(const double* targets, const double* scores, uint64_t size, double& value, double* grad) const = 0;
  };

}}

#endif // BOB_VISIONER_LOSS_H
