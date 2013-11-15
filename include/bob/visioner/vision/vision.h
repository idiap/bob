/**
 * @file bob/visioner/vision/vision.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_VISIONER_VISION_H
#define BOB_VISIONER_VISION_H

#include "bob/visioner/util/geom.h"
#include "bob/visioner/util/util.h"

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
