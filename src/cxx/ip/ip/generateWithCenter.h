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

    namespace detail {
      /**
        * @brief Update the output array by making the point with the given
        * coordinates on the input array/image, the center of the output 
        * (larger) array/image.
        * @param src The input array
        * @param src_mask The input blitz array mask, specifying the valid
        *   pixels of src.
        * @param dst The output array which should have the expected size. This 
        *   size can be obtained calling the getGenerateWithCenterShape() 
        *   function
        * @param dst_mask The input blitz array mask, specifying the valid
        *   pixels of dst.
        * @param target_center_h The y-coordinates of the point of the input 
        *   image which will becomes the center of the output array/image.
        * @param target_center_w The x-coordinates of the point of the input 
        *   image which will becomes the center of the output array/image.
        */
      template <typename T, bool mask>
      void generateWithCenterNoCheck( const blitz::Array<T,2>& src, 
        const blitz::Array<bool,2>& src_mask, blitz::Array<T,2>& dst,
        blitz::Array<bool,2>& dst_mask, const int target_center_h, 
        const int target_center_w)
      {
        // Compute offsets
        const blitz::TinyVector<int,2> offset =
          getGenerateWithCenterOffset(src,target_center_h,target_center_w);

        // Update output mask
        dst_mask = false;

        /// Update output content
        dst = 0.;
        blitz::Range src_h(0,src.extent(0)-1);
        blitz::Range src_w(0,src.extent(1)-1);
        blitz::Range dst1_h(offset(0),offset(0)+src.extent(0)-1);
        blitz::Range dst1_w(offset(1),offset(1)+src.extent(1)-1);
        dst(dst1_h,dst1_w) = src(src_h,src_w);

        // Output mask
        if(mask)
          dst_mask(dst1_h,dst1_w) = src_mask(src_h,src_w);
      }
    }

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
      * @brief Return the offsets, such that using these offsets, we get
      * the initial image on the new shifted image. The center of the new 
      * (larger) image will be the point with the given coordinates on the 
      * input array/image.
      */
    template <typename T>
    const blitz::TinyVector<int,2> getGenerateWithCenterOffset(
      const blitz::Array<T,2>& src, const int target_center_h, 
      const int target_center_w)
    {
      // Compute current center coordinates
      const double src_center_h = (src.extent(0)-1) / 2.;
      const double src_center_w = (src.extent(1)-1) / 2.;

      // Compute the extra boundaries needed
      blitz::TinyVector<int,2> res;
      res(0) = (target_center_h>=src_center_h ? 0 : 
        floor(2*fabs(src_center_h-target_center_h)) );
      res(1) = (target_center_w>=src_center_w ? 0 : 
        floor(2*fabs(src_center_w-target_center_w)) );

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

      // Compute dst 
      blitz::Array<bool,2> src_mask, dst_mask;
      detail::generateWithCenterNoCheck<T,false>( src, src_mask, dst, 
        dst_mask, target_center_h, target_center_w);
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
      const blitz::Array<bool,2>& src_mask, blitz::Array<T,2>& dst,
      blitz::Array<bool,2>& dst_mask, const int target_center_h, 
      const int target_center_w)
    {
      // Check input
      tca::assertZeroBase(src);
      tca::assertZeroBase(src_mask);

      // Check output
      tca::assertZeroBase(dst);
      tca::assertZeroBase(dst_mask);
      tca::assertSameShape(dst, dst_mask);
      tca::assertSameShape(dst,
        getGenerateWithCenterShape(src,target_center_h,target_center_w));

      // Compute dst
      detail::generateWithCenterNoCheck<T,true>( src, src_mask, dst, 
        dst_mask, target_center_h, target_center_w);
    }

	}
	/**
	 * @}
	 */
}

#endif // TORCH5SPRO_IP_GENERATE_WITH_CENTER_H
