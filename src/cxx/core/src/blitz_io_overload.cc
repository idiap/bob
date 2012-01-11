/**
 * @file cxx/core/src/blitz_io_overload.cc
 * @date Mon Apr 11 10:29:29 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file overloads the Input/Output stream operations on blitz++
 *   multidimensional arrays.
 * http://www.oonumerics.org/blitz/
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

#include "core/blitz_io_overload.h"

BZ_NAMESPACE(blitz)
// TODO
#ifdef BOB_NEEDS_REVISION

/** 
 * @brief specialization of operator << for int8_t in 1D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,1>& x)
{
  return out1D_template<int8_t,int16_t>(os,x);
}

/** 
 * @brief specialization of operator << for int8_t in 2D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,2>& x)
{
  return out2D_template<int8_t,int16_t>(os,x);
}

/**
 * @brief specialization of operator << for int8_t in 3D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,3>& x)
{
  return out3D_template<int8_t,int16_t>(os,x);
}

/**
 * @brief specialization of operator << for int8_t in 4D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,4>& x)
{
  return out4D_template<int8_t,int16_t>(os,x);
}


/** 
 * @brief specialization of operator << for uint8_t in 1D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,1>& x)
{
  return out1D_template<uint8_t,uint16_t>(os,x);
}

/** 
 * @brief specialization of operator << for uint8_t in 2D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,2>& x)
{
  return out2D_template<uint8_t,uint16_t>(os,x);
}

/**
 * @brief specialization of operator << for uint8_t in 3D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,3>& x)
{
  return out3D_template<uint8_t,uint16_t>(os,x);
}

/**
 * @brief specialization of operator << for uint8_t in 4D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,4>& x)
{
  return out4D_template<uint8_t,uint16_t>(os,x);
}

#endif
BZ_NAMESPACE_END
