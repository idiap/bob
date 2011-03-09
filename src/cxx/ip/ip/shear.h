/**
 * @file src/cxx/ip/ip/shear.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to shear/skew a 2D or 3D array/image.
 * The algorithm is strongly inspired by the following article:
 * 'A Fast Algorithm for General Raster Rotation', Alan Paeth, in the 
 * proceedings of Graphics Interface '86, p. 77-81.
 * The notes of Tobin Fricke about this article might also be of interest.
 */

#ifndef TORCH5SPRO_IP_SHEAR_H
#define TORCH5SPRO_IP_SHEAR_H 1

#include "core/logging.h"
#include "ip/Exception.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which copies a 2D blitz::array/image of a given type.
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        */
      template<typename T>
      void copyNoCheck(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst)
      { 
        for( int y=0; y<dst.extent(0); ++y)
          for( int x=0; x<dst.extent(1); ++x)
            dst(y,x) = src( y+src.lbound(0), x+src.lbound(1) );
      }

      
      /**
        * @brief Function which shears a 2D blitz::array/image of a given type
        *   along the X-axis.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        * @param shear The shear parameter in the matrix [1 shear; 0 1]
        */
      template<typename T>
      void shearXNoCheck(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst,
        const double shear, const bool antialias)
      {
        // Compute center coordinates in src and dst image
        double y_c = (src.extent(0) - 1)/ 2.;
        double x_c_src = (src.extent(1) - 1)/ 2.;
        double x_c_dst = (dst.extent(1) - 1)/ 2.;

        // If shear is equal to zero, we just need to do a simple copy
        if(shear == 0.) {
          copyNoCheck(src,dst);
          return;
        }

        // Initialize dst to background value
        dst = 0;

        // Loop over the rows and skew them horizontally
        for( int y=0; y<src.extent(0); ++y) {
          // Determine the skew offset wrt. the center of the input
          double skew = shear * (y - y_c);
          // Determine the direction and make the skew positive
          bool dir_right;
          if( skew > 0.)
            dir_right = true;
          else {
            dir_right = false;
            skew = -skew;
          }
          // Compute the floor of the skew
          int skew_i;
          if( antialias)
            skew_i = floor(skew);
          else
            skew_i = floor(skew+0.5);
          // Compute the residual of the skew
          double skew_f = skew - skew_i;
          double old_residual = 0.;

          // Transfer pixels right-to-left
          if( dir_right) {
            // Loop over all the input pixels of the row
            for( int x=src.extent(1)-1; x>=0; --x) {
              double pixel = src(y+src.lbound(0),x+src.lbound(1));
              double residual;
              if( antialias )
                residual = pixel * skew_f;
              else 
                residual = 0.;
              pixel = (pixel - residual) + old_residual;
              // Determine x-location on dst row
              int x_dst = floor(x - x_c_src + x_c_dst - skew_i+0.5);
              if( x_dst >= 0 && x_dst < dst.extent(1) )
                dst(y,x_dst) = (T)pixel; //TODO: check C-like cast
              old_residual = residual;
            }
            // Add remaining residual if possible
            double next_ind = -x_c_src + x_c_dst - skew_i - 1 + 0.5;
            if( next_ind >= 0)
              dst(y,(int)next_ind) = (T)old_residual; //TODO: check C-like cast
          }
          // Transfer pixels left-to-right
          else {
            // Loop over all the input pixels of the row
            for( int x=0; x<src.extent(1); ++x) {
              double pixel = src(y+src.lbound(0),x+src.lbound(1));
              double residual;
              if( antialias )
                residual = pixel * skew_f;
              else 
                residual = 0.;
              pixel = (pixel - residual) + old_residual;
              int x_dst = ceil(x - x_c_src + x_c_dst + skew_i-0.5);
              if( x_dst >= 0 && x_dst < dst.extent(1) )
                dst(y,x_dst) = (T)pixel; //TODO: check C-like cast
              old_residual = residual;
            }
            // Add remaining residual if possible
            double next_ind = 
              -x_c_src + x_c_dst + skew_i + src.extent(1) + 0.5;
            if( next_ind < dst.extent(1))
              dst(y,(int)next_ind) = (T)old_residual; //TODO: check C-like cast
          } 
        }    
      }

    }


    /**
      * @brief Function which shears a 2D blitz::array/image of a given type
      *   along the X-axis.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param shear The shear parameter in the matrix [1 shear; 0 1]
      * @param antialias Whether antialiasing should be used or not 
      */
    template<typename T>
    void shearX(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const double shear, const bool antialias=true)
    {
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        dst.reindexSelf( zero_base );
      }

      // Determine the new width of the image
      int width_dst = src.extent(1) + 
        floor(fabs(shear)*(src.extent(0)-1)+0.5);
      // Check and resize if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != width_dst)
        dst.resize( src.extent(0), width_dst );

      // Call the shearXNoCheck function
      detail::shearXNoCheck( src, dst, shear, antialias);
    }


    /**
      * @brief Function which shears a 2D blitz::array/image of a given type
      *   along the Y-axis.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param shear The shear parameter in the matrix [1 0; shear 1]
      * @param antialias Whether antialiasing should be used or not 
      */
    template<typename T>
    void shearY(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const double shear, const bool antialias=true)
    {
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        dst.reindexSelf( zero_base );
      }

      // Determine the new width of the image
      int height_dst = src.extent(0) + 
        floor(fabs(shear)*(src.extent(1)-1)+0.5);
      // Check and resize if required
      if( dst.extent(0) != height_dst || dst.extent(1) != src.extent(1))
        dst.resize( height_dst, src.extent(1) );

      // Create transposed view arrays for both src and dst
      const blitz::Array<T,2> src_transpose(src.transpose(1,0));
      blitz::Array<T,2> dst_transpose(dst.transpose(1,0));

      // Call the shearXNoCheck function
      detail::shearXNoCheck( src_transpose, dst_transpose, shear, antialias);
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_SHEAR_H */

