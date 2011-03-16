/**
 * @file src/cxx/core/core/common.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines common functions for the Torch library
 * 
 */

#ifndef TORCH5SPRO_CORE_COMMON_H
#define TORCH5SPRO_CORE_COMMON_H 1

#include "core/logging.h"

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    /**
      * @brief Force value to stay in a given range [min, max]
      * @param val The value to be considered
      * @param min The minimum of the range
      * @param max The maximum of the range
      */
    inline int keepInRange( const int val, const int min, const int max) {
      return (val < min ? min : (val > max ? max : val ) );
    }
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_CORE_COMMON_H */
