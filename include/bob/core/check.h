/**
 * @file bob/core/check.h
 * @date Wed Feb 9 12:26:11 2013 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#ifndef BOB_CORE_CHECK_H
#define BOB_CORE_CHECK_H

#include <cmath>
#include <algorithm>
#include <complex>
#include <blitz/array.h>

#include <bob/config.h>
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif

namespace bob {
/**
 * \ingroup libcore_api
 * @{
 *
 */
namespace core {

/**
 * @brief Compare two floating point values using two different comparison 
 * criteria: a relative one (for large values) and an absolute one (for small
 * values).
 * |left - right| <= (a_epsilon + r_epsilon * min(left,right))
 */
template<typename T> 
bool isClose(const T& left, const T& right, const T& r_epsilon=1e-5, 
  const T& a_epsilon=1e-8) 
{
  T diff = std::fabs(left - right);
  T min = std::min(left, right);
  return (diff <= (a_epsilon + r_epsilon * min));
}

/**
 * @brief Compare two complex values using two different comparison 
 * criteria: a relative one (for large values) and an absolute one (for small
 * values).
 * |left_ - right_| <= (a_epsilon + r_epsilon * min(left_,right_))
 *   where left_, right_ are real or imaginary parts of the arguments
 */
template<typename T> 
bool isClose(const std::complex<T>& left, const std::complex<T>& right, 
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8) 
{
  // Check real parts first
  T diff = std::fabs(left.real() - right.real());
  T min = std::min(left.real(), right.real());
  if (!(diff <= (a_epsilon + r_epsilon * min))) return false;
  // Check imaginary parts
  diff = std::fabs(left.imag() - right.imag());
  min = std::min(left.imag(), right.imag());
  return (diff <= (a_epsilon + r_epsilon * min));
}


namespace array {

/**
 * @brief Checks if a blitz array has zero base indices, for each of its 
 * dimensions. This is the case with default C-style order blitz arrays.
 */
template <typename T, int D>
bool isZeroBase( const blitz::Array<T,D>& a)
{
  for( int i=0; i<a.rank(); ++i)
    if( a.base(i)!=0 )
      return false;
  return true;
}

/**
 * @brief Checks if a blitz array has one base indices, for each of its
 * dimensions. This is the case with default Fortran order blitz arrays.
 */
template <typename T, int D>
bool isOneBase( const blitz::Array<T,D>& a)
{
  for( int i=0; i<a.rank(); ++i)
    if( a.base(i)!=1 )
      return false;
  return true;
}

/**
 * @brief Checks that one blitz array has the same base indices as another
 * blitz array.
 */
template <typename T, typename U, int D>
bool hasSameBase( const blitz::Array<T,D>& a, 
  const blitz::Array<U,D>& b)
{
  for( int i=0; i<D; ++i)
    if( a.base(i) != b.base(i) )
      return false;
  return true;
}

/**
 * @brief Checks if a blitz array is a C-style array stored in a 
 * contiguous memory area. By C-style array, it is meant that it is a row-
 * major order multidimensional array, i.e. strides are decreasing in size
 * as rank grows, and that the data is stored in ascending order in each
 * dimension.
 */
template <typename T, int D>
bool isCContiguous( const blitz::Array<T,D>& a) 
{
  if( !a.isStorageContiguous() )
    return false;
  for( int i=0; i<a.rank(); ++i) 
    if( !(a.isRankStoredAscending(i) && a.ordering(i)==a.rank()-1-i) )
      return false;
  return true;
}

/**
 * @brief Checks if a blitz array is a Fortran-style array stored in a 
 * contiguous memory area. By Fortran-style array, it is meant that it is 
 * a column-major order multidimensional array, i.e. strides are 
 * increasing in size as rank grows, and that the data is stored in 
 * ascending order in each dimension.
 */
template <typename T, int D>
bool isFortranContiguous( const blitz::Array<T,D>& a) 
{
  if( !a.isStorageContiguous() )
    return false;
  for( int i=0; i<a.rank(); ++i) 
    if( !(a.isRankStoredAscending(i) && a.ordering(i)==i) )
      return false;
  return true;
}

/**
 * @brief Checks if a blitz array is a C-style array stored in a 
 * contiguous memory area, with zero base indices. 
 */
template <typename T, int D>
bool isCZeroBaseContiguous( const blitz::Array<T,D>& a) 
{
  if( !isZeroBase(a) )
    return false;
  return isCContiguous(a);
}

/**
 * @brief Checks if a blitz array is a Fortran-style array stored in a 
 * contiguous memory area, with one base indices. 
 */
template <typename T, int D>
bool isFortranOneBaseContiguous( const blitz::Array<T,D>& a) 
{
  if( !isOneBase(a) )
    return false;
  return isFortranContiguous(a);
}

/**
 * @brief Checks that a blitz array has the same shape as the one
 * given in the second argument.
 */
template <typename T, int D>
bool hasSameShape( const blitz::Array<T,D>& ar, 
  const blitz::TinyVector<int, D>& shape)
{
  const blitz::TinyVector<int, D>& ar_shape = ar.shape();
  for( int i=0; i<D; ++i)
    if( ar_shape(i) != shape(i) )
      return false;
  return true;
}


/**
 * @brief Checks that a blitz array has the same shape as the one
 * given in the first argument.
 */
template <typename T, int D>
bool hasSameShape( const blitz::TinyVector<int, D>& shape, 
  const blitz::Array<T,D>& ar)
{
  const blitz::TinyVector<int, D>& ar_shape = ar.shape();
  for( int i=0; i<D; ++i)
    if( ar_shape(i) != shape(i) )
      return false;
  return true;
}


/**
 * @brief Checks that two blitz arrays have the same shape.
 */
template <typename T, typename U, int D>
bool hasSameShape( const blitz::Array<T,D>& a, 
  const blitz::Array<U,D>& b)
{
  const blitz::TinyVector<int, D>& a_shape = a.shape();
  const blitz::TinyVector<int, D>& b_shape = b.shape();
  for( int i=0; i<D; ++i)
    if( a_shape(i) != b_shape(i) )
      return false;
  return true;
}

/**
 * @brief Checks that two (floating point) 1D blitz arrays are close.
 */
template<typename T> 
bool isClose(const blitz::Array<T,1>& left, const blitz::Array<T,1>& right,
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8) 
{
  if (!hasSameShape(left, right))
    return false;
  
  for (int i=0; i<left.extent(0); ++i)
    if (!bob::core::isClose(left(i), right(i), r_epsilon, a_epsilon))
      return false;

  return true;
}

/**
 * @brief Checks that two (floating point) 2D blitz arrays are close.
 */
template<typename T> 
bool isClose(const blitz::Array<T,2>& left, const blitz::Array<T,2>& right, 
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8) 
{
  if (!hasSameShape(left, right))
    return false;
  
  for (int i=0; i<left.extent(0); ++i)
    for (int j=0; j<left.extent(1); ++j)
      if (!bob::core::isClose(left(i,j), right(i,j), r_epsilon, a_epsilon))
        return false;

  return true;
}

/**
 * @brief Checks that two (floating point) 3D blitz arrays are close.
 */
template<typename T> 
bool isClose(const blitz::Array<T,3>& left, const blitz::Array<T,3>& right,
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8)
{ 
  if (!hasSameShape(left, right))
    return false;
  
  for (int i=0; i<left.extent(0); ++i)
    for (int j=0; j<left.extent(1); ++j)
      for (int k=0; k<left.extent(2); ++k)
        if (!bob::core::isClose(left(i,j,k), right(i,j,k), r_epsilon, a_epsilon))
          return false;

  return true;
}

/**
 * @brief Checks that two (floating point) 4D blitz arrays are close.
 */
template<typename T> 
bool isClose(const blitz::Array<T,4>& left, const blitz::Array<T,4>& right,
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8)
{
  if (!hasSameShape(left, right))
    return false;
  
  for (int i=0; i<left.extent(0); ++i)
    for (int j=0; j<left.extent(1); ++j)
      for (int k=0; k<left.extent(2); ++k)
        for (int l=0; l<left.extent(3); ++l)
          if (!bob::core::isClose(left(i,j,k,l), right(i,j,k,l), r_epsilon, a_epsilon))
            return false;

  return true;
}

/**
 * @brief Checks that two (complex floating point) 1D blitz arrays are close.
 */
template<typename T> 
bool isClose(const blitz::Array<std::complex<T>,1>& left, 
  const blitz::Array<std::complex<T>,1>& right,
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8) 
{
  if (!hasSameShape(left, right))
    return false;
  
  for (int i=0; i<left.extent(0); ++i)
    if (!bob::core::isClose(left(i), right(i), r_epsilon, a_epsilon))
      return false;

  return true;
}

/**
 * @brief Checks that two (complex floating point) 2D blitz arrays are close.
 */
template<typename T> 
bool isClose(const blitz::Array<std::complex<T>,2>& left, 
  const blitz::Array<std::complex<T>,2>& right,
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8) 
{
  if (!hasSameShape(left, right))
    return false;
  
  for (int i=0; i<left.extent(0); ++i)
    for (int j=0; j<left.extent(1); ++j)
      if (!bob::core::isClose(left(i,j), right(i,j), r_epsilon, a_epsilon))
        return false;

  return true;
}

/**
 * @brief Checks that two (complex floating point) 3D blitz arrays are close.
 */
template<typename T> 
bool isClose(const blitz::Array<std::complex<T>,3>& left, 
  const blitz::Array<std::complex<T>,3>& right,
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8) 
{ 
  if (!hasSameShape(left, right))
    return false;
  
  for (int i=0; i<left.extent(0); ++i)
    for (int j=0; j<left.extent(1); ++j)
      for (int k=0; k<left.extent(2); ++k)
        if (!bob::core::isClose(left(i,j,k), right(i,j,k), r_epsilon, a_epsilon))
          return false;

  return true;
}

/**
 * @brief Checks that two (complex floating point) 4D blitz arrays are close.
 */
template<typename T> 
bool isClose(const blitz::Array<std::complex<T>,4>& left, 
  const blitz::Array<std::complex<T>,4>& right,
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8) 
{
  if (!hasSameShape(left, right))
    return false;
  
  for (int i=0; i<left.extent(0); ++i)
    for (int j=0; j<left.extent(1); ++j)
      for (int k=0; k<left.extent(2); ++k)
        for (int l=0; l<left.extent(3); ++l)
          if (!bob::core::isClose(left(i,j,k,l), right(i,j,k,l), r_epsilon, a_epsilon))
            return false;

  return true;
}

}}
/**
 * @}
 */
}

#endif /* BOB_CORE_CHECK_H */
