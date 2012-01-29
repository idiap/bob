/**
 * @file cxx/core/core/cast.h
 * @date Wed Feb 9 12:26:11 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines functions which add std::complex support to the
 * static_cast function.
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

#ifndef BOB_CORE_CAST_H
#define BOB_CORE_CAST_H

#include <blitz/array.h>
#include <stdint.h>
#include <complex>

namespace bob {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    /**
     * @brief Functions which add std::complex support to the static_cast
     * function. This is done by considering the real part only of any
     * complex number.
     */
    template<typename T, typename U> 
    T cast(const U& in) {
      return static_cast<T>(in);
    }


    /**
      * @brief Specializations of the cast function for the std::complex type.
      */
    // Complex to regular
    #define COMPLEX_TO_REGULAR_DECL(COMP, REG) template<> \
      REG cast<REG, COMP>( const COMP& in); 

    #define COMPLEX_TO_REGULAR_FULL_DECL(COMP) \
      COMPLEX_TO_REGULAR_DECL(COMP, bool) \
      COMPLEX_TO_REGULAR_DECL(COMP, int8_t) \
      COMPLEX_TO_REGULAR_DECL(COMP, int16_t) \
      COMPLEX_TO_REGULAR_DECL(COMP, int32_t) \
      COMPLEX_TO_REGULAR_DECL(COMP, int64_t) \
      COMPLEX_TO_REGULAR_DECL(COMP, uint8_t) \
      COMPLEX_TO_REGULAR_DECL(COMP, uint16_t) \
      COMPLEX_TO_REGULAR_DECL(COMP, uint32_t) \
      COMPLEX_TO_REGULAR_DECL(COMP, uint64_t) \
      COMPLEX_TO_REGULAR_DECL(COMP, float) \
      COMPLEX_TO_REGULAR_DECL(COMP, double) \
      COMPLEX_TO_REGULAR_DECL(COMP, long double) 

    COMPLEX_TO_REGULAR_FULL_DECL(std::complex<float>)
    COMPLEX_TO_REGULAR_FULL_DECL(std::complex<double>)
    COMPLEX_TO_REGULAR_FULL_DECL(std::complex<long double>)


    // Complex to complex
    #define COMPLEX_TO_COMPLEX_DECL(FROM, TO) template<> \
      TO cast<TO, FROM>( const FROM& in);

    #define COMPLEX_TO_COMPLEX_FULL_DECL(COMP) \
      COMPLEX_TO_COMPLEX_DECL(COMP, std::complex<float>) \
      COMPLEX_TO_COMPLEX_DECL(COMP, std::complex<double>) \
      COMPLEX_TO_COMPLEX_DECL(COMP, std::complex<long double>)

    COMPLEX_TO_COMPLEX_FULL_DECL(std::complex<float>)
    COMPLEX_TO_COMPLEX_FULL_DECL(std::complex<double>)
    COMPLEX_TO_COMPLEX_FULL_DECL(std::complex<long double>)


/**
 * @brief Casts a blitz array allowing std::complex types.
 */
/*
template<typename T, typename U, int D> 
Array<T,D> complex_cast(const Array<U,D>& in) {
  return cast<T>(in);
}
*/

template<typename T, typename U> 
blitz::Array<T,1> cast(const blitz::Array<U,1>& in) {
  blitz::Array<T,1> out(in.extent(0));
  for( int i=0; i<in.extent(0); ++i)
    out(i) = cast<T>( in(i+in.lbound(0)));
  return out;
}

template<typename T, typename U> 
blitz::Array<T,2> cast(const blitz::Array<U,2>& in) {
  blitz::Array<T,2> out(in.extent(0),in.extent(1));
  for( int i=0; i<in.extent(0); ++i)
    for( int j=0; j<in.extent(1); ++j)
      out(i,j) = cast<T>( in(i+in.lbound(0),j+in.lbound(1)) );
  return out;
}

template<typename T, typename U> 
blitz::Array<T,3> cast(const blitz::Array<U,3>& in) {
  blitz::Array<T,3> out(in.extent(0),in.extent(1),in.extent(2));
  for( int i=0; i<in.extent(0); ++i)
    for( int j=0; j<in.extent(1); ++j)
      for( int k=0; k<in.extent(2); ++k)
        out(i,j,k) = cast<T>( in(i+in.lbound(0),j+in.lbound(1),k+in.lbound(2)) );
  return out;
}

template<typename T, typename U> 
blitz::Array<T,4> cast(const blitz::Array<U,4>& in) {
  blitz::Array<T,4> out(in.extent(0),in.extent(1),in.extent(2),in.extent(3));
  for( int i=0; i<in.extent(0); ++i)
    for( int j=0; j<in.extent(1); ++j)
      for( int k=0; k<in.extent(2); ++k)
        for( int l=0; l<in.extent(3); ++l)
          out(i,j,k,l) = cast<T>( in(i+in.lbound(0),j+in.lbound(1),k+in.lbound(2),l+in.lbound(3)) );
  return out;
}





  }
/**
 * @}
 */
}

#endif /* BOB_CORE_CAST_H */

