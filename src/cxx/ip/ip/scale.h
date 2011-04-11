/**
 * @file src/cxx/ip/ip/scale.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to rescale a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_SCALE_H
#define TORCH5SPRO_IP_SCALE_H

#include "core/array_index.h"
#include "ip/Exception.h"
#include "ip/crop.h"

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
        * @brief Function which rescales a 2D blitz::array/image of a given 
        *   type, using bilinear interpolation.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        * @param height The height of the dst blitz::array.
        * @param width The width of the dst blitz::array.
        */
      template<typename T>
      void scaleNoCheck2D_BI(const blitz::Array<T,2>& src, 
        blitz::Array<T,2>& dst)
      {
        const int height = dst.extent(0);
        const int width = dst.extent(1);

        const double x_ratio = (src.extent(1)-1.) / (width-1.);
        const double y_ratio = (src.extent(0)-1.) / (height-1.);
        for( int y=0; y<height; ++y) {
          double y_src = y_ratio * y;
          double dy1 = ceil(y_src) - y_src;
          double dy2 = 1 - dy1;
          int y_ind1 = tca::keepInRange( floor(y_src), 0, src.extent(0)-1);
          int y_ind2 = tca::keepInRange( y_ind1+1, 0, src.extent(0)-1);
          for( int x=0; x<width; ++x) {
            double x_src = x_ratio * x;
            double dx1 = ceil(x_src) - x_src;
            double dx2 = 1 - dx1;
            int x_ind1 = tca::keepInRange( floor(x_src), 0, src.extent(1)-1);
            int x_ind2 = tca::keepInRange( x_ind1+1, 0, src.extent(1)-1);
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
      * @param src The input blitz array
      * @param dst The output blitz array. The new array is resized according
      *   to the dimensions of this dst array.
      * @param alg The algorithm used for rescaling.
      */
    template<typename T>
    void scale(const blitz::Array<T,2>& src, blitz::Array<T,2>& dst, 
      const enum Rescale::Algorithm alg=Rescale::BilinearInterp)
    {
      // Check and resize src if required
      tca::assertZeroBase(src);

      // Check and resize dst if required
      tca::assertZeroBase(dst);

      const int height = dst.extent(0);
      const int width = dst.extent(1);

      // Check parameters and throw exception if required
      if( width<1 ) {
        throw ParamOutOfBoundaryError("width", false, width, 1);
      }
      else if( height<0 ) {
        throw ParamOutOfBoundaryError("height", false, height, 1);
      }
  
      // TODO: if same dimension, make a simple copy
    
      // Rescale the 2D array
      switch(alg)
      {
        case Rescale::BilinearInterp:
          {
            // Rescale using Bilinear Interpolation
            detail::scaleNoCheck2D_BI<T>(src, dst);
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
