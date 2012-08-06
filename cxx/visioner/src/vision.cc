/**
 * @file visioner/src/vision.cc
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

#include "visioner/vision/vision.h"

namespace bob { namespace visioner {

  float distance(const QPointF& point1, const QPointF& point2)
  {
    const float diffx = point1.x() - point2.x();
    const float diffy = point1.y() - point2.y();
    return my_sqrt(diffx * diffx + diffy * diffy);
  }

  template <typename T>
    T square_euclidean(T a, T b)
    {
      return a * a + b * b;
    }

  // Jesorsky distance
  double Jesorsky(const QPointF& dt_leye, const QPointF& dt_reye,
      const QPointF& gt_leye, const QPointF& gt_reye)
  {
    const double d1 = square_euclidean(dt_leye.x() - gt_leye.x(), dt_leye.y() - gt_leye.y());	
    const double d2 = square_euclidean(dt_reye.x() - gt_reye.x(), dt_reye.y() - gt_reye.y());
    const double EE = square_euclidean(gt_reye.x() - gt_leye.x(), gt_reye.y() - gt_reye.y());

    return my_sqrt((d1 > d2 ? d1 : d2) / EE);
  }

  double Jesorsky(const QPointF& dt_leye, const QPointF& dt_reye,
      const std::vector<QPointF>& gt_leyes, const std::vector<QPointF>& gt_reyes)
  {
    double dist = 1.0;
    for (std::vector<QPointF>::const_iterator itl = gt_leyes.begin(), itr = gt_reyes.begin();
        itl != gt_leyes.end(); ++ itl, ++ itr)
    {
      dist = std::min(dist, Jesorsky(dt_leye, dt_reye, *itl, *itr));
    }
    return dist;
  }

  // Overlapping [0.0 - 1.0]
  double overlap(const QRectF& det, const std::vector<QRectF>& gts)
  {
    double over = 0.0;
    for (std::vector<QRectF>::const_iterator it = gts.begin(); it != gts.end(); ++ it)
    {
      over = std::max(over, overlap(det, *it));
    }
    return over;
  }

}}
