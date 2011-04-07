/**
 * @file src/cxx/ip/ip/shiftToCenter.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @author <a href="mailto:Niklas Johansson@idiap.ch">Niklas Johansson</a> 
 *
 * @brief This file defines a function to shift 2D or 3D blitz array to
 * a center point.
 * 
 */

#ifndef TORCH5SPRO_IP_SHIFT_TO_CENTER_H
#define TORCH5SPRO_IP_SHIFT_TO_CENTER_H

#include "core/logging.h"
#include "ip/Exception.h"
#include "ip/common.h"
#include "shift.h"

namespace Torch {
	/**
	 * \ingroup libip_api
	 * @{
	 *
	 */
	namespace ip {

		template<typename T>
		void shiftToCenter(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst,
				   const int target_center_h, const int target_center_w)
		{
			const int delta_h = target_center_h - src.extent(0) / 2;
			const int delta_w = target_center_w - src.extent(1) / 2;

			shift(src, dst, delta_h, delta_w);
		}

		template<typename T>
		void shiftToCenterOfPoints(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst,
					   const int point_one_h, const int point_one_w,
					   const int point_two_h, const int point_two_w)
		{
			const int target_center_h = (point_one_h + point_two_h) / 2;
			const int target_center_w = (point_one_w + point_two_w) / 2;

			shiftToCenter(src, dst, target_center_h, target_center_w);
		}

		template<typename T>
		void shiftToCenter(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst,
				   const int target_center_h, const int target_center_w)
		{
			const int delta_h = target_center_h - src.extent(1) / 2;
			const int delta_w = target_center_w - src.extent(2) / 2;
      
			shift(src, dst, delta_h, delta_w);
		}

		template<typename T>
		void shiftToCenterOfPoints(const blitz::Array<T,3>& src, blitz::Array<T,3>& dst,
					   const int point_one_h, const int point_one_w,
					   const int point_two_h, const int point_two_w)
		{
			const int target_center_h = (point_one_h + point_two_h) / 2;
			const int target_center_w = (point_one_w + point_two_w) / 2;

			shiftToCenter(src, dst, target_center_h, target_center_w);
		}
	}

	/**
	 * @}
	 */
}

#endif // TORCH5SPRO_IP_SHIFT_TO_CENTER_H
