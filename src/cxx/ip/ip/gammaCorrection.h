/**
 * @file src/cxx/ip/ip/gammaCorrection.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to perform power-law gamma correction 
 *   on a 2D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_GAMMA_CORRECTION_H
#define TORCH5SPRO_IP_GAMMA_CORRECTION_H 1

#include "core/logging.h"
#include "ip/Exception.h"
#include "ip/common.h"
#include <cmath>

namespace Torch {
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
        blitz::Range src_y( src.lbound(0), src.ubound(0) ),
                     src_x( src.lbound(1), src.ubound(1) );
        blitz::Range dst_y( dst.lbound(0), dst.ubound(0) ),
                     dst_x( dst.lbound(1), dst.ubound(1) );
        dst(dst_y,dst_x) = pow( src(src_y,src_x), gamma);
      }

    }


    /**
      * @brief Function which performs a gamma correction on a 2D 
      *   blitz::array/image of a given type.
      *   The first dimension is the height (y-axis), whereas the second
      *   one is the width (x-axis).
      * @warning The dst blitz::array/image is resized and reindexed with zero
      *   base index.
      * @param src The input blitz array
      * @param dst The output blitz array (always double)
      * @param gamma The gamma value for power-law gamma correction
      */
    template<typename T>
    void gammaCorrection(const blitz::Array<T,2>& src, 
      blitz::Array<double,2>& dst, const double gamma)
    {
      // Check and reindex if required
      if( dst.base(0) != 0 || dst.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        dst.reindexSelf( zero_base );
      }
      // Check and resize dst if required
      if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) )
        dst.resize( src.extent(0), src.extent(1) );

      // Check parameters and throw exception if required
      if( gamma < 0.) 
      {
        throw ParamOutOfBoundaryError("gamma", false, gamma, 0.);
      }
    
      // Perform gamma correction for the 2D array
      detail::gammaCorrectionNoCheck(src, dst, gamma);
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_GAMMA_CORRECTION_H */
