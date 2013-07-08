/**
 * @file bob/core/assert.h
 * @date Sat Apr 9 18:10:10 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines assert functions over the blitz++ arrays
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

#ifndef BOB_CORE_ASSERT_H
#define BOB_CORE_ASSERT_H

#include <stdexcept>
#include <boost/format.hpp>
#include <bob/core/check.h>

namespace bob {
namespace core { namespace array {
/**
 * @ingroup CORE_ARRAY
 * @{
 */
namespace detail {
  template<typename T, int D>
  std::string tinyvec2str(const blitz::TinyVector<T,D>& tv) {
    std::ostringstream oss;
    oss << "[";
    for (int i=0; i<(D-1); ++i) oss << tv(i) << ",";
    oss << tv(D-1) << "]";
    return oss.str();
  }
}

/**
 * @brief Checks that a blitz array has zero base indices, and throws
 * a std::runtime_error exception if this is not the case.
 */
template<typename T, int D>
void assertZeroBase(const blitz::Array<T,D>& src)
{
  for( int i=0; i<src.rank(); ++i)
    if( src.base(i)!=0 ) {
      boost::format m("input array has dimension %d with a non-zero base index (base=%d)");
      m % i % src.base(i); 
      throw std::runtime_error(m.str());
    }
}

/**
 * @brief Checks that a blitz array has one base indices, and throws
 * a std::runtime_error exception if this is not the case.
 */
template<typename T, int D>
void assertOneBase(const blitz::Array<T,D>& src)
{
  for( int i=0; i<src.rank(); ++i)
    if( src.base(i)!=1) {
      boost::format m("input array has dimension %d with a non-one base index (base=%d)");
      m % i % src.base(i); 
      throw std::runtime_error(m.str());
    }
}

/**
 * @brief Checks that two blitz arrays have the same base, and throws a
 * std::runtime_error exception if this is not the case.
 */
template<typename T, typename U, int D>
void assertSameBase(const blitz::Array<T,D>& a, 
  const blitz::Array<U,D>& b)
{
  if( !hasSameBase(a,b) )
    throw std::runtime_error("arrays do not have the same base");
}

/**
 * @brief Checks that a blitz array is a C-style array stored contiguously
 * in memory, and throws a std::runtime_error exception if this is not 
 * the case.
 */
template<typename T, int D>
void assertCContiguous(const blitz::Array<T,D>& src)
{
  if( !isCContiguous(src) ) {
    throw std::runtime_error("array is not C-style (row-major order) and stored in a continguous memory area");
  }
}

/**
 * @brief Checks that a blitz array is a Fortran-style array stored 
 * contiguously in memory, and throws a std::runtime_error exception if 
 * this is not the case.
 */
template<typename T, int D>
void assertFortranContiguous(const blitz::Array<T,D>& src)
{
  if( !isFortranContiguous(src) ) {
    throw std::runtime_error("array is not fortran-style (column-major order) and stored in a continguous memory area");
  }
}

/**
 * @brief Checks that a blitz array is a C-style array stored contiguously
 * in memory with zero base indices, and throws a 
 * std::runtime_error exception if this is not the case.
 */
template<typename T, int D>
void assertCZeroBaseContiguous(const blitz::Array<T,D>& src)
{
  assertZeroBase(src);
  assertCContiguous(src);
}

/**
 * @brief Checks that a blitz array is a Fortran-style array stored 
 * contiguously in memory with one base indices, and throws a 
 * std::runtime_error exception if this 
 * is not the case.
 */
template<typename T, int D>
void assertFortranOneBaseContiguous(const blitz::Array<T,D>& src)
{
  assertOneBase(src);
  assertFortranContiguous(src);
}

/**
 * @brief Checks that a blitz array has the expected shape, and throws an
 * std::runtime_error exception if this is not the case.
 */
template<typename T, int D>
void assertSameShape(const blitz::Array<T,D>& ar, 
  const blitz::TinyVector<int, D>& shape)
{
  if( !hasSameShape(ar,shape) ) {
    boost::format m("array shape %s does not match expected value %s");
    m % detail::tinyvec2str(ar.shape()) % detail::tinyvec2str(shape);
    throw std::runtime_error(m.str());
  }
}

/**
 * @brief Checks that a blitz array has the expected shape, and throws an
 * std::runtime_error exception if this is not the case.
 */
template<typename T, int D>
void assertSameShape(const blitz::TinyVector<int,D>& shape, 
  const blitz::Array<T,D>& ar)
{
  if( !hasSameShape(shape,ar) ) {
    boost::format m("array shape %s does not match expected value %s");
    m % detail::tinyvec2str(ar.shape()) % detail::tinyvec2str(shape);
    throw std::runtime_error(m.str());
  }
}

/**
 * @brief Checks that two blitz arrays have the same shape, and throws an
 * std::runtime_error exception if this is not the case.
 */
template<typename T, typename U, int D>
void assertSameShape(const blitz::Array<T,D>& a, 
  const blitz::Array<U,D>& b)
{
  if( !hasSameShape(a,b) ) {
    boost::format m("array shapes do not match %s != %s");
    m % detail::tinyvec2str(a.shape()) % detail::tinyvec2str(b.shape());
    throw std::runtime_error(m.str());
  }
}

/**
 * @brief Checks that two dimensions (values) have the same length (value),
 * and throws an std::runtime_error exception if this is not the case.
 */
inline void assertSameDimensionLength(const int d1, const int d2)
{
  if( d1!=d2 ) {
    boost::format m("array dimensions do not match %d != %d");
    m % d1 % d2;
    throw std::runtime_error(m.str());
  }
}

/**
 * @}
 */
}}}

#endif /* BOB_ARRAY_ASSERT_H */
