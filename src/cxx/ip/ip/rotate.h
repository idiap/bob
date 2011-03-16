/**
 * @file src/cxx/ip/ip/rotate.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to rotate a 2D or 3D array/image.
 * The algorithm is strongly inspired by the following article:
 * 'A Fast Algorithm for General Raster Rotation', Alan Paeth, in the 
 * proceedings of Graphics Interface '86, p. 77-81.
 * The notes of Tobin Fricke about this article might also be of interest.
 */

#ifndef TORCH5SPRO_IP_ROTATE_H
#define TORCH5SPRO_IP_ROTATE_H 1

#include "core/logging.h"
#include "ip/Exception.h"
#include "ip/common.h"
#include "ip/shear.h"
#include "ip/crop.h"

#include <iostream>

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which rotates a 2D blitz::array/image of a given type
        *   with an angle of 90 degrees.
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        */
      template<typename T>
      void rotateNoCheck_90(const blitz::Array<T,2>& src, 
        blitz::Array<T,2>& dst)
      { 
        for( int y=0; y<dst.extent(0); ++y)
          for( int x=0; x<dst.extent(1); ++x)
            dst(y,x) = src( x+src.lbound(0), (src.extent(1)-1-y)+src.lbound(1) );
      }

      /**
        * @brief Function which rotates a 2D blitz::array/image of a given type
        *   with an angle of 180 degrees.
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        */
      template<typename T>
      void rotateNoCheck_180(const blitz::Array<T,2>& src, 
        blitz::Array<T,2>& dst)
      { 
        for( int y=0; y<dst.extent(0); ++y)
          for( int x=0; x<dst.extent(1); ++x)
            dst(y,x) = src( (src.extent(0)-1-y)+src.lbound(0), (src.extent(1)-1-x)+src.lbound(1) );
      }

      /**
        * @brief Function which rotates a 2D blitz::array/image of a given type
        *   with an angle of 270 degrees.
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        */
      template<typename T>
      void rotateNoCheck_270(const blitz::Array<T,2>& src, 
        blitz::Array<T,2>& dst)
      { 
        for( int y=0; y<dst.extent(0); ++y)
          for( int x=0; x<dst.extent(1); ++x)
            dst(y,x) = src( (src.extent(0)-1-x)+src.lbound(0), y+src.lbound(1) );
      }

    }

    namespace Rotation {
      enum Algorithm {
        Shearing,
        BilinearInterp
      };
    }

    /**
      * @brief Function which rotates a 2D blitz::array/image of a given type.
      *   The center of the rotation is the center of the image.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param angle The angle of the rotation (in degrees)
      * @param alg The algorithm which should be used to perform the rotation.
      */
    template<typename T>
    void rotate(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const double angle, 
      const enum Rotation::Algorithm alg=Rotation::Shearing)
    {
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        dst.reindexSelf( zero_base );
      }
  
      // Force the angle to be in range [-45,45] and determine the quadrant
      // of the original angle:
      //   0:[-45,45] -- 1:[45,135] -- 2:[135,225] -- 3:[225,315(i.e. -45)]
      double angle_norm = angle;
      size_t quadrant;
      while(angle_norm < -45.)
        angle_norm += 360.;

      // Check and resize dst if required
      if(angle_norm == 0. || angle_norm == 180.)
      {
        // Resize output
        if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) )
          dst.resize( src.extent(0), src.extent(1) );
        // Perform rotation
        if(angle_norm == 0.)
          detail::copyNoCheck(src, dst);
        else
          detail::rotateNoCheck_180(src, dst);
        return;
      }
      else if(angle_norm == 90. || angle_norm == 270. || angle_norm == -45.)
      {
        // Resize output
        if( dst.extent(0) != src.extent(1) || dst.extent(1) != src.extent(0) )
          dst.resize( src.extent(1), src.extent(0) );
        // Perform rotation
        if(angle_norm == 90.)
          detail::rotateNoCheck_90(src, dst);
        else
          detail::rotateNoCheck_270(src, dst);
        return;
      }

      for( quadrant=0; angle_norm > 45.; ++quadrant)
        angle_norm -= 90.;
      quadrant %= 4;

      // Compute useful values 
      double rad_angle = angle_norm * M_PI / 180.;

      switch(alg)
      {
        case Rotation::Shearing:
          {
            // Declare an intermediate arrays
            blitz::Array<T,2> dst_int1, dst_int2;;
            // Perform simple rotation. After that, there is one more rotation 
            // to do with an angle in [-45,45].
            if( quadrant == 0 ) {
              dst_int1.reference( src );
            }
            else if( quadrant == 1) {
              dst_int1.resize( src.extent(1), src.extent(0) );
              detail::rotateNoCheck_90(src, dst_int1);  
            }
            else if( quadrant == 2) {
              dst_int1.resize( src.extent(0), src.extent(1) );
              detail::rotateNoCheck_180(src, dst_int1);  
            }
            else { // quadrant == 3
              dst_int1.resize( src.extent(1), src.extent(0) );
              detail::rotateNoCheck_270(src, dst_int1);  
            }

            // Compute shearing values required for the rotation
            const double shear_x = -tan( rad_angle / 2. );
            const double shear_y = sin( rad_angle );
  
            // Performs first shear (shearX)
            shearX( dst_int1, dst_int2, shear_x, true);
            // Performs second shear (shearY)
            shearY( dst_int2, dst_int1, shear_y, true);
            // Performs third shear (shearX)
            shearX( dst_int1, dst_int2, shear_x, true);

            // Crop obtained sheared image
            const double dAbsSin = fabs(sin(rad_angle));
            const double dAbsCos = fabs(cos(rad_angle));
            const int dst_width = 
              floor(src.extent(0)*dAbsSin + src.extent(1)*dAbsCos + 0.5);
            const int dst_height = 
              floor(src.extent(0)*dAbsCos + src.extent(1)*dAbsSin + 0.5);
      
            int crop_x = (dst_int2.extent(1) - dst_width) / 2;
            int crop_y = (dst_int2.extent(0) - dst_height) / 2;
            crop( dst_int2, dst, crop_x, crop_y, dst_width, dst_height, true,
              true);
          }
          break;
        default:
          throw Torch::ip::Exception();
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_ROTATE_H */
