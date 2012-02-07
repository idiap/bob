/**
 * @file cxx/core/core/array_index.h
 * @date Mon Apr 11 10:29:29 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file contains functions related to array indices manipulation
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_CORE_ARRAY_INDEX_H
#define BOB_CORE_ARRAY_INDEX_H

#include <stdint.h>
#include <cstdlib>
#include <complex>

namespace bob {
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

#endif /* BOB_CORE_ARRAY_INDEX_H */
