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
      * @brief Force value to stay in a given range [min, max]. In case of out
      *   of range values, the closest value is returned (i.e. min or max)
      * @param val The value to be considered
      * @param min The minimum of the range
      * @param max The maximum of the range
      */
    inline int keepInRange( const int val, const int min, const int max) {
      return (val < min ? min : (val > max ? max : val ) );
    }

    /**
      * @brief Force value to stay in a given range [min, max]. In case of out
      *   of range values, 'mirroring' is performed. For instance:
      *     mirrorInRange(-1, 0, 5) will return 0.
      *     mirrorInRange(-2, 0, 5) will return 1.
      *     mirrorInRange(17, 3, 15) will return 14.
      * @param val The value to be considered
      * @param min The minimum of the range
      * @param max The maximum of the range
      */
    inline int mirrorInRange( const int val, const int min, const int max) {
      return (val < min ? mirrorInRange(min-val-1, min, max) : 
                (val > max ? mirrorInRange(2*max-val+1, min, max) : val ) );
    }
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_CORE_COMMON_H */
