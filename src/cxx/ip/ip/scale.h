/**
 * @file src/cxx/ip/ip/scale.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to rescale a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_SCALE_H
#define TORCH5SPRO_IP_SCALE_H 1

#include "core/logging.h"
#include "ip/Exception.h"
#include "ip/crop.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which rescales a 2D blitz::array/image of a given 
        *   type, using bilinear interpolation.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        * @param width The width of the dst blitz::array.
        * @param height The height of the dst blitz::array.
        */
      template<typename T>
      void scaleNoCheck2D_BI(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst,
        const int width, const int height)
      {
        const double x_ratio = (src.extent(1)-1.) / (width-1.);
        const double y_ratio = (src.extent(0)-1.) / (height-1.);
        for( int y=0; y<height; ++y) {
          double y_src = y_ratio * y;
          double dy1 = ceil(y_src) - y_src;
          double dy2 = 1 - dy1;
          int y_ind1 = Torch::core::keepInRange( floor(y_src), 0, src.extent(0)-1);
          int y_ind2 = Torch::core::keepInRange( y_ind1+1, 0, src.extent(0)-1);
          for( int x=0; x<width; ++x) {
            double x_src = x_ratio * x;
            double dx1 = ceil(x_src) - x_src;
            double dx2 = 1 - dx1;
            int x_ind1 = Torch::core::keepInRange( floor(x_src), 0, src.extent(1)-1);
            int x_ind2 = Torch::core::keepInRange( x_ind1+1, 0, src.extent(1)-1);
            double val = dx1*dy1*src(y_ind1, x_ind1)+dx1*dy2*src(y_ind2, x_ind1)
              + dx2*dy1*src(y_ind1, x_ind2 )+dx2*dy2*src(y_ind2, x_ind2 );
            dst(y,x) = (T)val; // TODO Check C-style cast
          }
        }
      }

    }

    namespace Rescale {
      enum Algorithm {
        NearestNeighbour,
        BilinearInterp
      };
    }

    /**
      * @brief Function which rescales a 2D blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param width The width of the dst blitz::array.
      * @param height The height of the dst blitz::array.
      * @param alg The algorithm used for rescaling.
      */
    template<typename T>
    void scale(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const int width, const int height, 
      const enum Rescale::Algorithm alg=Rescale::BilinearInterp)
    {
      // Check and resize dst if required
      if( dst.extent(0) != height || dst.extent(1) != width )
        dst.resize( height, width );
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        dst.reindexSelf( zero_base );
      }

      // Check parameters and throw exception if required
      if( width<0 || height<0 ) 
      {
        if( width<0 ) {
          throw ParamOutOfBoundaryError("width", false, width, 0);
        }
        else if( height<0 ) {
          throw ParamOutOfBoundaryError("height", false, height, 0);
        }
        else
          throw Exception();
      }
  
      // TODO: if same dimension, make a simple copy
    
      // Rescale the 2D array
      switch(alg)
      {
        case Rescale::BilinearInterp:
          {
            // Rescale using Bilinear Interpolation
            detail::scaleNoCheck2D_BI<T>(src, dst, width, height);
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

#endif /* TORCH5SPRO_IP_SCALE_H */

