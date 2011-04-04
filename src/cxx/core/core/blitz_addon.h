/**
 * @file src/cxx/core/core/blitz_addon.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file contains miscellaneous additions for blitz++ arrays
 *
 * http://www.oonumerics.org/blitz/
 */

#ifndef TORCH_CORE_BLITZ_ADDON_H
#define TORCH_CORE_BLITZ_ADDON_H

#include <blitz/array.h>

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

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
     * @brief Checks if a blitz array is a C-style array stored in a 
     * contiguous memory area. By C-style array, it is meant that it is a row-
     * major order multidimensional array, i.e. strides are decreasing in size
     * as rank grows, and that the data is stored in ascending order in each
     * dimension.
     */
    template <typename T, int D>
    inline bool isCContiguous( const blitz::Array<T,D>& a) 
    {
      if( !a.isStorageContiguous() )
        return false;
      // TODO: Check that there is no need to look at the stride in the blitz
      // implementation
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
    inline bool isFortranContiguous( const blitz::Array<T,D>& a) 
    {
      if( !a.isStorageContiguous() )
        return false;
      // TODO: Check that there is no need to look at the stride in the blitz
      // implementation
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
    inline bool isCZeroBaseContiguous( const blitz::Array<T,D>& a) 
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
    inline bool isFortranOneBaseContiguous( const blitz::Array<T,D>& a) 
    {
      if( !isOneBase(a) )
        return false;
      return isFortranContiguous(a);
    }


    /**
     * @brief This function reindex and resize a 1D blitz array with the given
     * parameters
     * @param array The 1D blitz array to reindex and resize
     * @param base0 The base index of the first dimension
     * @param size0 The size of the first dimension
     * @warning If a resizing is performed, previous content of the array is 
     * lost.
     */
    template <typename T>
    void reindexAndResize( blitz::Array<T,1>& array, const int base0, 
      const int size0)
    {
      // Check and reindex if required
      if( array.base(0) != base0) {
        const blitz::TinyVector<int,1> base( base0);
        array.reindexSelf( base );
      }
      // Check and resize if required
      if( array.extent(0) != size0)
        array.resize( size0);
    }

    /**
     * @brief This function reindex and resize a 2D blitz array with the given
     * parameters
     * @param array The 2D blitz array to reindex and resize
     * @param base0 The base index of the first dimension
     * @param base1 The base index of the second dimension
     * @param size0 The size of the first dimension
     * @param size1 The size of the second dimension
     * @warning If a resizing is performed, previous content of the array is 
     * lost.
     */
    template <typename T>
    void reindexAndResize( blitz::Array<T,2>& array, const int base0, 
      const int base1, const int size0, const int size1)
    {
      // Check and reindex if required
      if( array.base(0) != base0 || array.base(1) != base1) {
        const blitz::TinyVector<int,2> base( base0, base1);
        array.reindexSelf( base );
      }
      // Check and resize if required
      if( array.extent(0) != size0 || array.extent(1) != size1)
        array.resize( size0, size1);
    }

    /**
     * @brief This function reindex and resize a 3D blitz array with the given
     * parameters
     * @param array The 3D blitz array to reindex and resize
     * @param base0 The base index of the first dimension
     * @param base1 The base index of the second dimension
     * @param size0 The size of the first dimension
     * @param size1 The size of the second dimension
     * @warning If a resizing is performed, previous content of the array is 
     * lost.
     */
    template <typename T>
    void reindexAndResize( blitz::Array<T,3>& array, const int base0, 
      const int base1, const int base2, const int size0, const int size1, 
      const int size2)
    {
      // Check and reindex if required
      if( array.base(0) != base0 || array.base(1) != base1 || 
        array.base(2) != base2) 
      {
        const blitz::TinyVector<int,3> base( base0, base1, base2);
        array.reindexSelf( base );
      }
      // Check and resize if required
      if( array.extent(0) != size0 || array.extent(1) != size1 || 
          array.extent(2) != size2)
        array.resize( size0, size1, size2);
    }

    /**
     * @brief This function reindex and resize a 4D blitz array with the given
     * parameters
     * @param array The 4D blitz array to reindex and resize
     * @param base0 The base index of the first dimension
     * @param base1 The base index of the second dimension
     * @param base2 The base index of the third dimension
     * @param size0 The size of the first dimension
     * @param size1 The size of the second dimension
     * @param size2 The size of the third dimension
     * @warning If a resizing is performed, previous content of the array is 
     * lost.
     */
    template <typename T>
    void reindexAndResize( blitz::Array<T,4>& array, const int base0,
      const int base1, const int base2, const int base3, const int size0, 
      const int size1, const int size2, const int size3)
    {
      // Check and reindex if required
      if( array.base(0) != base0 || array.base(1) != base1 || 
        array.base(2) != base2 || array.base(3) != base3) 
      {
        const blitz::TinyVector<int,3> base( base0, base1, base2, base3);
        array.reindexSelf( base );
      }
      // Check and resize if required
      if( array.extent(0) != size0 || array.extent(1) != size1 || 
          array.extent(2) != size2 || array.extent(3) != size3)
        array.resize( size0, size1, size2, size3);
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_CORE_BLITZ_ADDON_H */
