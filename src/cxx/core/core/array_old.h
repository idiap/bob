/**
 * @file cxx/core/core/array_old.h
 * @date Mon Apr 11 10:29:29 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file contains deprecated checks and copy functions for arrays
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

#ifndef BOB_CORE_ARRAY_OLD_H
#define BOB_CORE_ARRAY_OLD_H

#include <cstdlib>
#include <blitz/array.h>

BZ_NAMESPACE(blitz)

/**
 * @brief This function check that the data() function of a 1D blitz array
 * can be used safely, i.e.:
 *   * the memory is stored contiguously
 *   * data is not reversed in each dimension
 *   * Row major storage order is used
 */
template <typename T>
inline bool checkSafedata( const Array<T,1>& src) 
{
  if( src.isStorageContiguous() && src.isRankStoredAscending(0) && 
    (src.stride(0) == 1) )
    return true;
  return false;
}

/**
 * @brief This function check that the data() function of a 2D blitz array
 * can be used safely, i.e.:
 *   * the memory is stored contiguously
 *   * data is not reversed in each dimension
 *   * Row major storage order is used
 */
template <typename T>
inline bool checkSafedata( const Array<T,2>& src) 
{
  if( src.isStorageContiguous() && src.isRankStoredAscending(0) && 
    src.isRankStoredAscending(1) && (src.ordering(0)==1) && 
    (src.ordering(1)==0) && (src.stride(0) == src.extent(1)) && 
    (src.stride(1) == 1))
    return true;
  return false;
}

/**
 * @brief This function check that the data() function of a 3D blitz array
 * can be used safely, i.e.:
 *   * the memory is stored contiguously
 *   * data is not reversed in each dimension
 *   * Row major storage order is used
 */
template <typename T>
inline bool checkSafedata( const Array<T,3>& src) 
{
  if( src.isStorageContiguous() && src.isRankStoredAscending(0) && 
    src.isRankStoredAscending(1) && src.isRankStoredAscending(2) && 
    (src.ordering(0)==2) && (src.ordering(1)==1) && (src.ordering(2)==0) &&
    (src.stride(0) == src.extent(1)*src.extent(2)) && 
    (src.stride(1) == src.extent(2)) && (src.stride(2) == 1))
    return true;
  return false;
}

/**
 * @brief This function check that the data() function of a 4D blitz array
 * can be used safely, i.e.:
 *   * the memory is stored contiguously
 *   * data is not reversed in each dimension
 *   * Row major storage order is used
 */
template <typename T>
inline bool checkSafedata( const Array<T,4>& src) 
{
  if( src.isStorageContiguous() && src.isRankStoredAscending(0) && 
    src.isRankStoredAscending(1) && src.isRankStoredAscending(2) && 
    src.isRankStoredAscending(3) && (src.ordering(0)==3) && 
    (src.ordering(1)==2) && (src.ordering(2)==1) && (src.ordering(3)==0) &&
    (src.stride(0) == src.extent(1)*src.extent(2)*src.extent(3)) && 
    (src.stride(1) == src.extent(2)*src.extent(3)) && 
    (src.stride(2) == src.extent(3)) && (src.stride(3) == 1))
    return true;
  return false;
}


/**
 * @brief This copies a 1D blitz array and guaranties that:
 *   * the memory is stored contiguously
 *   * indices start at 0 and data are not reversed in each dimension
 *   * Row major storage order is used
 * @warning If you want to use the output of this function, you can use:
 *   * The blitz copy constructor
 *   * The reference() function
 * Please note than using the assignment will require to do a full copy.
 */
template <typename T> 
Array<T,1> copySafedata( const Array<T,1>& src)
{
  // Create dst array
  Array<T,1> dst( src.extent(0) );

  // Make a (safe) copy
  Range r_src( src.lbound(0), src.ubound(0) ),
        r_dst( dst.lbound(0), dst.ubound(0) ); 
  dst(r_dst) = src(r_src);
  
  return dst;
}

/**
 * @brief This copies a 2D blitz array and guaranties that:
 *   * the memory is stored contiguously
 *   * indices start at 0 and data are not reversed in each dimension
 *   * Row major storage order is used
 */
template <typename T> 
Array<T,2> copySafedata( const Array<T,2>& src)
{
  // Create dst array
  Array<T,2> dst( src.extent(0), src.extent(1) );

  // Make a (safe) copy
  Range r_src0( src.lbound(0), src.ubound(0) ),
        r_src1( src.lbound(1), src.ubound(1) ),
        r_dst0( dst.lbound(0), dst.ubound(0) ),
        r_dst1( dst.lbound(1), dst.ubound(1) ); 
  dst(r_dst0,r_dst1) = src(r_src0,r_src1);

  return dst;
}

/**
 * @brief This copies a 3D blitz array and guaranties that:
 *   * the memory is stored contiguously
 *   * indices start at 0 and data are not reversed in each dimension
 *   * Row major storage order is used
 */
template <typename T> 
Array<T,3> copySafedata( const Array<T,3>& src)
{
  // Create dst array
  Array<T,3> dst( src.extent(0), src.extent(1), src.extent(2) );

  // Make a (safe) copy
  Range r_src0( src.lbound(0), src.ubound(0) ),
        r_src1( src.lbound(1), src.ubound(1) ),
        r_src2( src.lbound(2), src.ubound(2) ),
        r_dst0( dst.lbound(0), dst.ubound(0) ),
        r_dst1( dst.lbound(1), dst.ubound(1) ), 
        r_dst2( dst.lbound(2), dst.ubound(2) ); 
  dst(r_dst0,r_dst1,r_dst2) = src(r_src0,r_src1,r_src2);

  return dst;
}

/**
 * @brief This copies a 4D blitz array and guaranties that:
 *   * the memory is stored contiguously
 *   * indices start at 0 and data are not reversed in each dimension
 *   * Row major storage order is used
 */
template <typename T> 
Array<T,4> copySafedata( const Array<T,4>& src)
{
  // Create dst array
  Array<T,4> dst( src.extent(0), src.extent(1), src.extent(2), src.exent(3) );

  // Make a (safe) copy
  Range r_src0( src.lbound(0), src.ubound(0) ),
        r_src1( src.lbound(1), src.ubound(1) ),
        r_src2( src.lbound(2), src.ubound(2) ),
        r_src3( src.lbound(3), src.ubound(3) ),
        r_dst0( dst.lbound(0), dst.ubound(0) ),
        r_dst1( dst.lbound(1), dst.ubound(1) ), 
        r_dst2( dst.lbound(2), dst.ubound(2) ),
        r_dst3( dst.lbound(3), dst.ubound(3) );
  dst(r_dst0,r_dst1,r_dst2,r_dst3) = src(r_src0,r_src1,r_src2,r_src3);

  return dst;
}

BZ_NAMESPACE_END

#endif /* BOB_CORE_ARRAY_OLD_H */
