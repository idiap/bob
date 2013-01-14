/**
 * @file core/cxx/cast.cc
 * @date Wed Feb 9 16:19:18 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines functions which add std::complex support to the
 * static_cast function.
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

#include "bob/core/cast.h"

/**
  * @brief Specializations of the cast function for the std::complex type.
  */
// Complex to regular
#define COMPLEX_TO_REGULAR(COMP, REG) template<> \
  REG bob::core::cast<REG, COMP>( const COMP& in) \
  { \
    return static_cast<REG>(in.real()); \
  }
  
#define COMPLEX_TO_REGULAR_FULL(COMP) \
  COMPLEX_TO_REGULAR(COMP, bool) \
  COMPLEX_TO_REGULAR(COMP, int8_t) \
  COMPLEX_TO_REGULAR(COMP, int16_t) \
  COMPLEX_TO_REGULAR(COMP, int32_t) \
  COMPLEX_TO_REGULAR(COMP, int64_t) \
  COMPLEX_TO_REGULAR(COMP, uint8_t) \
  COMPLEX_TO_REGULAR(COMP, uint16_t) \
  COMPLEX_TO_REGULAR(COMP, uint32_t) \
  COMPLEX_TO_REGULAR(COMP, uint64_t) \
  COMPLEX_TO_REGULAR(COMP, float) \
  COMPLEX_TO_REGULAR(COMP, double) \
  COMPLEX_TO_REGULAR(COMP, long double)

  COMPLEX_TO_REGULAR_FULL(std::complex<float>)
  COMPLEX_TO_REGULAR_FULL(std::complex<double>)
  COMPLEX_TO_REGULAR_FULL(std::complex<long double>)

// Complex to complex
#define COMPLEX_TO_COMPLEX(FROM, TO) template<> \
  TO bob::core::cast<TO, FROM>( const FROM& in) \
  { \
    return static_cast<TO>(in); \
  }

#define COMPLEX_TO_COMPLEX_FULL(COMP) \
  COMPLEX_TO_COMPLEX(COMP, std::complex<float>) \
  COMPLEX_TO_COMPLEX(COMP, std::complex<double>) \
  COMPLEX_TO_COMPLEX(COMP, std::complex<long double>) 

  COMPLEX_TO_COMPLEX_FULL(std::complex<float>)
  COMPLEX_TO_COMPLEX_FULL(std::complex<double>)
  COMPLEX_TO_COMPLEX_FULL(std::complex<long double>)

