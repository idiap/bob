/**
 * @file visioner/src/jesorsky_loss.cc
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

#include "bob/visioner/model/losses/jesorsky_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  double JesorskyLoss::error(
      const double* targets, const double* scores, uint64_t size) const
  {
    const uint64_t n_points = points(size);
    const double scale = 2.0 * inverse(eye_dist(targets));

    double sum_error = 0;           
    for (uint64_t p = 0; p < n_points; p ++)
    {
      const double dx = scores[2 * p + 0] - targets[2 * p + 0];
      const double dy = scores[2 * p + 1] - targets[2 * p + 1];
      sum_error += scale * std::sqrt(dx * dx + dy * dy);
    }

    return sum_error;
  }

  // Compute the loss value & derivatives
  void JesorskyLoss::eval(
      const double* targets, const double* scores, uint64_t size,
      double& value) const
  {
    value = 0.0;

    const uint64_t n_points = points(size);
    const double scale = 2.0 * inverse(eye_dist(targets));

    for (uint64_t p = 0; p < n_points; p ++)
    {
      const double dx = scores[2 * p + 0] - targets[2 * p + 0];
      const double dy = scores[2 * p + 1] - targets[2 * p + 1];
      value += scale * std::sqrt(dx * dx + dy * dy);
    }
  }
  void JesorskyLoss::eval(
      const double* targets, const double* scores, uint64_t size,
      double& value, double* grad) const
  {
    value = 0.0;

    const uint64_t n_points = points(size);
    const double scale = 2.0 * inverse(eye_dist(targets));

    for (uint64_t p = 0; p < n_points; p ++)
    {
      const double dx = scores[2 * p + 0] - targets[2 * p + 0];
      const double dy = scores[2 * p + 1] - targets[2 * p + 1];

      const double sq = std::sqrt(dx * dx + dy * dy);
      const double isq = inverse(sq);

      value += scale * sq;
      grad[2 * p + 0] = scale * isq * dx;
      grad[2 * p + 1] = scale * isq * dy;
    }
  }

}}
