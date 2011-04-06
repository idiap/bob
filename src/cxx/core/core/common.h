/**
 * @file src/cxx/core/core/common.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines common functions for the Torch library
 * 
 */

#ifndef TORCH5SPRO_CORE_COMMON_H
#define TORCH5SPRO_CORE_COMMON_H 1

#include "core/Exception.h"

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
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_CORE_COMMON_H */
