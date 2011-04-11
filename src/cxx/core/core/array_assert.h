/**
 * @file src/cxx/core/core/array_assert.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines assert functions over the blitz++ arrays
 */

#ifndef TORCH5SPRO_CORE_ARRAY_ASSERT_H
#define TORCH5SPRO_CORE_ARRAY_ASSERT_H

#include "core/array_check.h"
#include "core/array_exception.h"
#include <blitz/tinyvec-et.h>

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {
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
      if( !isSameShape(ar,shape) )
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
      if( !isSameShape(a,b) )
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

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_ARRAY_ASSERT_H */
