/**
 * @file src/cxx/ip/ip/geoNorm.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to geoNorm a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_GEONORM_H
#define TORCH5SPRO_IP_GEONORM_H 1

#include <cmath>
#include "core/logging.h"
#include "ip/Exception.h"
#include "core/common.h"
#include "ip/rotate.h"
#include "ip/scale.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    template<typename T>
    void geoNorm(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
		 const int left_h, const int left_w,
		 const int right_h, const int right_w,
		 const int target_distance_between_left_and_right,
		 const int final_height, const int final_width)
    {
	    //shift to center
	    blitz::Array<T,2> dst_cnt(src.shape());
	    shiftToCenterOfPoints(src, dst_cnt, left_h, left_w, right_h, right_w);
	    
	    // rotate
	    double angle = getRotateAngleToLevelOutHorizontal(left_h, left_w, right_h, right_w);
	    blitz::Array<T,2> dst_rot(getShapeRotated(dst_cnt, angle));
	    rotate(dst_cnt, dst_rot, angle);
	    
	    // scale the 2D tensor
	    // old distance between points vs target distance between points
	    const double old_distance = std::sqrt((right_h - left_h) * (right_h - left_h)
						  +
						  (right_w - left_w) * (right_w - left_w));
	    const double scale_factor = 1.0 * target_distance_between_left_and_right / old_distance;

	    const int new_height      = floor(src.extent(0) * scale_factor + 0.5);
	    const int new_width       = floor(src.extent(1) * scale_factor + 0.5);
	    
	    blitz::Array<T,2> dst_scl(new_height, new_width);
	    scale(dst_rot, dst_scl, new_height, new_width);

	    // crop around the center WARNING, TODO find correct formula for the cropping
	    blitz::Array<T,2> dst_crp(final_height, final_width);
	    cropAroundCenter(dst_scl, dst_crp, final_height, final_width, true, true);
    }

/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_GEONORM_H */

