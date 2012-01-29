/**
 * @file cxx/core/core/array_assert.h
 * @date Sat Apr 9 18:10:10 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines assert functions over the blitz++ arrays
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

#ifndef BOB_CORE_ARRAY_ASSERT_H
#define BOB_CORE_ARRAY_ASSERT_H

#include "core/array_check.h"
#include "core/array_exception.h"
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif

namespace bob {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core { namespace array {
    /**
     * @brief Checks that a blitz array has zero base indices, and throws
     * a NonZeroBaseError exception if this is not the case.
     */
    template<typename T, int D>
    void assertZeroBase(const blitz::Array<T,D>& src)
    {
      for( int i=0; i<src.rank(); ++i)
        if( src.base(i)!=0 )
          throw NonZeroBaseError( i, src.base(i));
    }

    /**
     * @brief Checks that a blitz array has one base indices, and throws
     * a NonOneBaseError exception if this is not the case.
     */
    template<typename T, int D>
    void assertOneBase(const blitz::Array<T,D>& src)
    {
      for( int i=0; i<src.rank(); ++i)
        if( src.base(i)!=1)
          throw NonOneBaseError( i, src.base(i));
    }

    /**
     * @brief Checks that two blitz arrays have the same base, and throws an
     * DifferentBaseError exception if this is not the case.
     */
    template<typename T, typename U, int D>
    void assertSameBase(const blitz::Array<T,D>& a, 
      const blitz::Array<U,D>& b)
    {
      if( !hasSameBase(a,b) )
        throw DifferentBaseError();
    }

    /**
     * @brief Checks that a blitz array is a C-style array stored contiguously
     * in memory, and throws a NonCContiguousError exception if this is not 
     * the case.
     */
    template<typename T, int D>
    void assertCContiguous(const blitz::Array<T,D>& src)
    {
      if( !isCContiguous(src) )
        throw NonCContiguousError();
    }

    /**
     * @brief Checks that a blitz array is a Fortran-style array stored 
     * contiguously in memory, and throws a NonCContiguousError exception if 
     * this is not the case.
     */
    template<typename T, int D>
    void assertFortranContiguous(const blitz::Array<T,D>& src)
    {
      if( !isFortranContiguous(src) )
        throw NonFortranContiguousError();
    }

    /**
     * @brief Checks that a blitz array is a C-style array stored contiguously
     * in memory with zero base indices, and throws a 
     * NonCContiguousError/NonZeroBaseError exception if this is not the case.
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
     * NonFortranContiguousError/NonZeroBaseError exception if this 
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
     * UnexpectedShapeError exception if this is not the case.
     */
    template<typename T, int D>
    void assertSameShape(const blitz::Array<T,D>& ar, 
      const blitz::TinyVector<int, D>& shape)
    {
      if( !hasSameShape(ar,shape) )
        throw UnexpectedShapeError();
    }

    /**
     * @brief Checks that two blitz arrays have the same shape, and throws an
     * UnexpectedShapeError exception if this is not the case.
     */
    template<typename T, typename U, int D>
    void assertSameShape(const blitz::Array<T,D>& a, 
      const blitz::Array<U,D>& b)
    {
      if( !hasSameShape(a,b) )
        throw UnexpectedShapeError();
    }

    /**
     * @brief Checks that two dimensions (values) have the same length (value),
     * and throws an UnexpectedShapeError exception if this is not the case.
     */
    inline void assertSameDimensionLength(const int d1, const int d2)
    {
      if( d1!=d2 )
        throw UnexpectedShapeError();
    }

  }}
/**
 * @}
 */
}

#endif /* BOB_ARRAY_ASSERT_H */
