/**
 * @file src/cxx/ip/ip/generateWithCenter.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @author <a href="mailto:Niklas Johansson@idiap.ch">Niklas Johansson</a> 
 *
 * @brief This file defines a function to generate a 2D/3D blitz array whose
 * center will correspond to a given point of the input array.
 * 
 */

#ifndef TORCH5SPRO_IP_GENERATE_WITH_CENTER_H
#define TORCH5SPRO_IP_GENERATE_WITH_CENTER_H

#include "ip/Exception.h"
#include "ip/shift.h"
#include "core/array_assert.h"

namespace tca = Torch::core::array;

namespace Torch {
	/**
	 * \ingroup libip_api
	 * @{
	 *
	 */
	namespace ip {

    /**
      * @brief Return the shape of an image, where the point with the given 
      * coordinates on the input array/image, will be the center on a new 
      * (larger) image.
      */
    template <typename T>
    const blitz::TinyVector<int,2> getGenerateWithCenterShape(
      const blitz::Array<T,2>& src, const int target_center_h, 
      const int target_center_w)
    {
      // Compute current center coordinates
      const double current_center_h = (src.extent(0)-1) / 2.;
      const double current_center_w = (src.extent(1)-1) / 2.;

      // Compute the extra boundaries needed
      blitz::TinyVector<int,2> res;
      res(0) = floor(2*fabs(target_center_h-current_center_h)) + src.extent(0);
      res(1) = floor(2*fabs(target_center_w-current_center_w)) + src.extent(1);

      return res;
    }

    /**
      * @brief Update the output array by making the point with the given
      * coordinates on the input array/image, the center of the output 
      * (larger) array/image.
      * @param src The input array
      * @param dst The output array which should have the expected size. This 
      *   size can be obtained calling the getGenerateWithCenterShape() 
      *   function
      * @param target_center_h The y-coordinates of the point of the input 
      *   image which will becomes the center of the output array/image.
      * @param target_center_w The x-coordinates of the point of the input 
      *   image which will becomes the center of the output array/image.
      */
    template <typename T>
    void generateWithCenter( const blitz::Array<T,2>& src, 
      blitz::Array<T,2>& dst, const int target_center_h, 
      const int target_center_w)
    {
      // Check input
      tca::assertZeroBase(src);

      // Check output
      tca::assertZeroBase(dst);
      tca::assertSameShape(dst,
        getGenerateWithCenterShape(src,target_center_h,target_center_w));

      // Compute center coordinates of the input
      const double src_center_h = (src.extent(0)-1) / 2.;
      const double src_center_w = (src.extent(1)-1) / 2.;
  
      // Compute offset in the output image
      const int h_offset = (target_center_h>=src_center_h ? 0 : 
        floor(2*fabs(src_center_h-target_center_h)) );
      const int w_offset = (target_center_w>=src_center_w ? 0 : 
        floor(2*fabs(src_center_w-target_center_w)) );

      /// Update output content
      dst = 0.;
      blitz::Range src_h(0,src.extent(0)-1);
      blitz::Range src_w(0,src.extent(1)-1);
      blitz::Range dst1_h(h_offset,h_offset+src.extent(0)-1);
      blitz::Range dst1_w(w_offset,w_offset+src.extent(1)-1);
      dst(dst1_h,dst1_w) = src(src_h,src_w);
    }


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
