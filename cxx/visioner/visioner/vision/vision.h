/**
 * @file visioner/visioner/vision/vision.h
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

#ifndef BOB_VISIONER_VISION_H
#define BOB_VISIONER_VISION_H

#include "visioner/util/geom.h"
#include "visioner/util/util.h"

namespace bob { namespace visioner {

  // 2D Euclidean distance
  float distance(const QPointF& point1, const QPointF& point2);

  // Jesorsky distance	
  double Jesorsky(	const QPointF& dt_leye, const QPointF& dt_reye,		// detection
      const QPointF& gt_leye, const QPointF& gt_reye);	// ground truth
  double Jesorsky(	const QPointF& dt_leye, const QPointF& dt_reye,		// detection
      const std::vector<QPointF>& gt_leyes, const std::vector<QPointF>& gt_reyes);	// ground truth

  // Overlapping [0.0 - 1.0]
  inline double overlap(const QRectF& det, const QRectF& gt)
  {
    const QRectF inter = det.intersected(gt);
    return area(inter) / (area(det) + area(gt) - area(inter));
  }	
  double overlap(const QRectF& det, const std::vector<QRectF>& gts);

}}

#endif // BOB_VISIONER_VISION_H
