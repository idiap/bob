#ifndef BOB_VISIONER_VISION_H
#define BOB_VISIONER_VISION_H

#include "visioner/util/geom.h"
#include "visioner/util/util.h"

namespace bob { namespace visioner {

  // Matrix used to store grayscale values
  typedef u_int8_t			grey_t;
  typedef Matrix<grey_t>			greyimage_t;
  typedef std::vector<greyimage_t>	greyimages_t;

  // Matrix used to store integral images
  typedef u_int32_t			igrey_t;
  typedef Matrix<igrey_t>			igreyimage_t;
  typedef std::vector<igreyimage_t>	igreyimages_t;

  // 2D Euclidean distance
  float distance(const point_t& point1, const point_t& point2);

  // Jesorsky distance	
  scalar_t Jesorsky(	const point_t& dt_leye, const point_t& dt_reye,		// detection
      const point_t& gt_leye, const point_t& gt_reye);	// ground truth
  scalar_t Jesorsky(	const point_t& dt_leye, const point_t& dt_reye,		// detection
      const points_t& gt_leyes, const points_t& gt_reyes);	// ground truth

  // Overlapping [0.0 - 1.0]
  inline scalar_t overlap(const rect_t& det, const rect_t& gt)
  {
    const rect_t inter = det.intersected(gt);
    return area(inter) / (area(det) + area(gt) - area(inter));
  }	
  scalar_t overlap(const rect_t& det, const rects_t& gts);

}}

#endif // BOB_VISIONER_VISION_H
