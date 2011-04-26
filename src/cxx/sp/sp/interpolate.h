/**
 * @file src/cxx/sp/sp/interpolate.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements various interpolation techniques for 1D and 2D blitz
 * arrays.
 */

#ifndef TORCH5SPRO_SP_INTERPOLATE_H
#define TORCH5SPRO_SP_INTERPOLATE_H

#include <blitz/array.h>

namespace Torch {
  /**
   * \ingroup libsp_api
   * @{
   *
   */
  namespace sp {

    namespace detail {
      /** 
       * @brief Perform bilinear interpolation in the 1D src array for the
       *   given floating point coordinates.
       *
       * @param src The input blitz array.
       * @param y y-coordinate of the point to interpolate
       * @param x x-coordinate of the point to interpolate
       */
      template<typename T> 
      double bilinearInterpolationNoCheck(const blitz::Array<T,1>& src, 
          const double y, const double x)
      {
        int xl = static_cast<int>(floor(x));
        int xh = static_cast<int>(ceil(x));

        return (xh-x)*src(xl) + (1-(xh-x))*src(xh);
      }

      /** 
       * @brief Perform bilinear interpolation in the 2D src array for the
       *   given floating point coordinates.
       *
       * @param src The input blitz array.
       * @param y y-coordinate of the point to interpolate
       * @param x x-coordinate of the point to interpolate
       */
      template<typename T> 
      double bilinearInterpolationNoCheck(const blitz::Array<T,2>& src, 
          const double y, const double x)
      {
        int yl = static_cast<int>(floor(y));
        int yh = static_cast<int>(ceil(y));
        int xl = static_cast<int>(floor(x));
        int xh = static_cast<int>(ceil(x));

        const double Il = (xh-x)*src(yl,xl) + (1-(xh-x))*src(yl,xh);
        const double Ih = (xh-x)*src(yh,xl) + (1-(xh-x))*src(yh,xh);

        return (yh-y)*Il + (1-(yh-y))*Ih;
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_INTERPOLATE_H */
