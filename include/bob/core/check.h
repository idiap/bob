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
#include <bob/core/array_check.h>

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
 */
template<typename T> 
bool isClose(const T& left, const T& right, const T& r_epsilon=1e-5, 
  const T& a_epsilon=1e-8) 
{
  T diff = std::fabs(left - right);
  T min = std::min(left, right);
  return (diff < (a_epsilon + r_epsilon * min));
}

/**
 * @brief Compare two complex values using two different comparison 
 * criteria: a relative one (for large values) and an absolute one (for small
 * values).
 */
template<typename T> 
bool isClose(const std::complex<T>& left, const std::complex<T>& right, 
  const T& r_epsilon=1e-5, const T& a_epsilon=1e-8) 
{
  // Check real parts first
  T diff = std::fabs(left.real() - right.real());
  T min = std::min(left.real(), right.real());
  if (!(diff < (a_epsilon + r_epsilon * min))) return false;
  // Check imaginary parts
  diff = std::fabs(left.imag() - right.imag());
  min = std::min(left.imag(), right.imag());
  return (diff < (a_epsilon + r_epsilon * min));
}

namespace array {

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
