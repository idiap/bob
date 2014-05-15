/**
 * @date Mon Apr 11 10:29:29 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file contains functions related to array indices manipulation
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_CORE_ARRAY_INDEX_H
#define BOB_CORE_ARRAY_INDEX_H

#include <cstdlib>

namespace bob { namespace core { namespace array {
/**
 * @ingroup CORE_ARRAY
 * @{
 */

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
 * @ingroup CORE_ARRAY
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

/**
 * @}
 */
}}}

#endif /* BOB_CORE_ARRAY_INDEX_H */
