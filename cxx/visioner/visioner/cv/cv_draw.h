#ifndef BOB_VISIONER_CV_DRAW_H
#define BOB_VISIONER_CV_DRAW_H

#include "visioner/cv/cv_detector.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Utility drawing functions:
  //      - ground truth objects
  //      - detections
  //      - keypoint localizations
  //      - predicted labels
  /////////////////////////////////////////////////////////////////////////////////////////

  QImage draw_gt(const ipscale_t& ipscale);
  void draw_detection(QImage& qimage, const detection_t& det, const param_t& param, bool label);
  void draw_detections(QImage& qimage, const detections_t& dets, const param_t& param, const bools_t& labels);
  void draw_points(QImage& qimage, const points_t& points);
  void draw_label(QImage& qimage, const detection_t& detection, const param_t& param, 
      index_t gt_label, index_t dt_label);

}}

#endif // BOB_VISIONER_CV_DRAW_H
