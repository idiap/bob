/**
 * @file cxx/sp/sp/interpolate.h
 * @date Mon Apr 25 18:06:14 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements various interpolation techniques for 1D and 2D blitz
 * arrays.
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BOB5SPRO_SP_INTERPOLATE_H
#define BOB5SPRO_SP_INTERPOLATE_H

#include <blitz/array.h>

namespace bob {
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

#endif /* BOB5SPRO_SP_INTERPOLATE_H */
