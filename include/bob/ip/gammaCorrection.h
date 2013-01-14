/**
 * @file bob/ip/gammaCorrection.h
 * @date Thu Mar 17 18:46:09 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to perform power-law gamma correction
 *   on a 2D array/image.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_IP_GAMMA_CORRECTION_H
#define BOB_IP_GAMMA_CORRECTION_H

#include <blitz/array.h>
#include <cmath>
#include "bob/core/array_assert.h"
#include "bob/ip/Exception.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
      /**
        * @brief Function which performs a gamma correction on a 2D 
        *   blitz::array/image of a given type.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        * @param gamma The gamma value for power-law gamma correction
        */
      template<typename T>
      void gammaCorrectionNoCheck(const blitz::Array<T,2>& src, 
        blitz::Array<double,2>& dst, const double gamma)
      {
        dst = blitz::pow( src, gamma);
      }

    }


    /**
      * @brief Function which performs a gamma correction on a 2D 
      *   blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @param src The input blitz array
      * @param dst The output blitz array (always double)
      * @param gamma The gamma value for power-law gamma correction
      */
    template<typename T>
    void gammaCorrection(const blitz::Array<T,2>& src, 
      blitz::Array<double,2>& dst, const double gamma)
    {
      // Check input/output
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertSameShape(dst, src); 

      // Check parameters and throw exception if required
      if( gamma < 0.) 
        throw ParamOutOfBoundaryError("gamma", false, gamma, 0.);
    
      // Perform gamma correction for the 2D array
      detail::gammaCorrectionNoCheck(src, dst, gamma);
    }

  }
/**
 * @}
 */
}

#endif /* BOB_IP_GAMMA_CORRECTION_H */
