/**
 * @file src/cxx/ip/ip/common.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines common functions for processing 2D/3D array image.
 * 
 */

#ifndef TORCH5SPRO_IP_COMMON_H
#define TORCH5SPRO_IP_COMMON_H 1

#include "core/logging.h"
#include "core/common.h"
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
      void copyNoCheck(const blitz::Array<T,2>& src, 
        blitz::Array<T,2>& dst)
      { 
        for( int y=0; y<dst.extent(0); ++y)
          for( int x=0; x<dst.extent(1); ++x)
            dst(y,x) = src( y+src.lbound(0), x+src.lbound(1) );
      }

      /**
        * @brief Function which copies a 3D blitz::array/image of a given type.
        * @warning No check is performed on the dst blitz::array/image.
        * @param src The input blitz array
        * @param dst The output blitz array
        */
      template<typename T>
      void copyNoCheck(const blitz::Array<T,3>& src, 
        blitz::Array<T,3>& dst)
      { 
        for( int p=0; p<dst.extent(0); ++p)
          for( int y=0; y<dst.extent(1); ++y)
            for( int x=0; x<dst.extent(2); ++x)
              dst(p,y,x) = src( p+src.lbound(0), y+src.lbound(1), 
                x+src.lbound(2) );
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_COMMON_H */

