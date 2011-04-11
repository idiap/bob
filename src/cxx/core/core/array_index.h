/**
 * @file src/cxx/core/core/array_index.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file contains functions related to array indices manipulation
 *
 */

#ifndef TORCH_CORE_ARRAY_INDEX_H
#define TORCH_CORE_ARRAY_INDEX_H

#include <stdint.h>
#include <cstdlib>
#include <complex>

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core { namespace array {

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

  }}
/**
 * @}
 */
}

#endif /* TORCH_CORE_ARRAY_INDEX_H */
