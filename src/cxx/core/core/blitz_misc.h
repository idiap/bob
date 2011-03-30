/**
 * @file src/cxx/core/core/blitz_misc.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file contains miscellaneous additions for blitz++ arrays
 *
 * http://www.oonumerics.org/blitz/
 */

#ifndef TORCH_CORE_BLITZ_MISC_H
#define TORCH_CORE_BLITZ_MISC_H

#include <blitz/array.h>

BZ_NAMESPACE(blitz)

/**
 * @brief This function reindex and resize a 1D blitz array with the given
 * parameters
 * @param array The 1D blitz array to reindex and resize
 * @param base0 The base index of the first dimension
 * @param size0 The size of the first dimension
 * @warning If a resizing is performed, previous content of the array is lost.
 */
template <typename T>
void reindexAndResize( Array<T,1>& array, const int base0, const int size0)
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
 * @warning If a resizing is performed, previous content of the array is lost.
 */
template <typename T>
void reindexAndResize( Array<T,2>& array, const int base0, const int base1, 
  const int size0, const int size1)
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
 * @warning If a resizing is performed, previous content of the array is lost.
 */
template <typename T>
void reindexAndResize( Array<T,3>& array, const int base0, const int base1, 
  const int base2, const int size0, const int size1, const int size2)
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
 * @warning If a resizing is performed, previous content of the array is lost.
 */
template <typename T>
void reindexAndResize( Array<T,4>& array, const int base0, const int base1,
  const int base2, const int base3, const int size0, const int size1, 
  const int size2, const int size3)
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


//TODO
#ifdef TORCH_NEEDS_REVISION
/**
 * @brief Local function for outputting a 1D int8/uint8 Blitz++ array,
 * by converting it to a printable type (int16/uint16)
 */
template <typename Tsrc, typename Tdst>
ostream& out1D_template(ostream& os, const Array<Tsrc,1>& x)
{
  os << "(" << x.lbound(firstRank) << "," << x.ubound(firstRank) << ")" << 
    endl;

  os << " [ ";
  for (int i=x.lbound(firstRank); i <= x.ubound(firstRank); ++i)
  {
    os << setw(9) << static_cast<Tdst>(x(i)) << " ";
    if (!((i+1-x.lbound(firstRank))%7))
      os << endl << "  ";
  }
  os << " ]";
  return os;
}

/**
 * @brief Local function for outputting a 2D int8/uint8 Blitz++ array,
 * by converting it to a printable type (int16/uint16)
 */
template <typename Tsrc, typename Tdst>
ostream& out2D_template(ostream& os, const Array<Tsrc,2>& x)
{
  os << "(" << x.lbound(firstRank) << "," << x.ubound(firstRank) << ")" << 
    " x (" << x.lbound(secondRank) << "," << x.ubound(secondRank) << ")" << 
    endl;

  os << "[ ";
  for (int i=x.lbound(firstRank); i <= x.ubound(firstRank); ++i)
  {
    for (int j=x.lbound(secondRank); j <= x.ubound(secondRank); ++j)
    {
      os << setw(9) << static_cast<Tdst>(x(i,j)) << " ";
      if (!((j+1-x.lbound(secondRank)) % 7))
        os << endl << "  ";
    }

    if (i != x.ubound(firstRank))
      os << endl << "  ";
  }

  os << "]" << endl;

  return os;
}

/**
 * @brief Local function for outputting a 3D int8/uint8 Blitz++ array,
 * by converting it to a printable type (int16/uint16)
 */
template <typename Tsrc, typename Tdst>
ostream& out3D_template(ostream& os, const Array<Tsrc,3>& x)
{
  os << "(" << x.lbound(firstRank) << "," << x.ubound(firstRank) << ")" << 
    " x (" << x.lbound(secondRank) << "," << x.ubound(secondRank) << ")" << 
    " x (" << x.lbound(thirdRank) << "," << x.ubound(thirdRank) << ")" << 
    std::endl;

  os << endl << "[ ";
  for (int i=x.lbound(firstRank); i <= x.ubound(firstRank); ++i) {
    for (int j=x.lbound(secondRank); j <= x.ubound(secondRank); ++j) {
      for (int k=x.lbound(thirdRank); k <= x.ubound(thirdRank); ++k) {
        os << setw(9) << static_cast<Tdst>(x(i,j,k)) << " ";
        if (!((k+1-x.lbound(thirdRank)) % 7))
          os << endl << "  ";
      }
      if (j != x.ubound(secondRank))
        os << endl << "  ";
    }
    if (i != x.ubound(firstRank))
      os << endl << endl << "  ";
  }
  os << endl << "]" << endl;

  return os;
}

/**
 * @brief Local function for outputting a 4D int8/uint8 Blitz++ array,
 * by converting it to a printable type (int16/uint16)
 */
template <typename Tsrc, typename Tdst>
ostream& out4D_template(ostream& os, const Array<Tsrc,4>& x)
{
  os << "(" << x.lbound(firstRank) << "," << x.ubound(firstRank) << ")" << 
    " x (" << x.lbound(secondRank) << "," << x.ubound(secondRank) << ")" << 
    " x (" << x.lbound(thirdRank) << "," << x.ubound(thirdRank) << ")" << 
    " x (" << x.lbound(fourthRank) << "," << x.ubound(fourthRank) << ")" << 
    std::endl;

  os << endl << "[ ";
  for (int i=x.lbound(firstRank); i <= x.ubound(firstRank); ++i) {
    for (int j=x.lbound(secondRank); j <= x.ubound(secondRank); ++j) {
      for (int k=x.lbound(thirdRank); k <= x.ubound(thirdRank); ++k) {
        for (int l=x.lbound(fourthRank); l <= x.ubound(fourthRank); ++l) {
          os << setw(9) << static_cast<Tdst>(x(i,j,k,l)) << " ";
          if (!((l+1-x.lbound(fourthRank)) % 7))
            os << endl << "  ";
        }
        if (k != x.ubound(thirdRank))
          os << endl << "  ";
      }
      if (j != x.ubound(secondRank))
        os << endl << endl << "  ";
    }
    if (i != x.ubound(firstRank))
      os << endl << endl << endl << "  ";
  }
  os << endl << "]" << endl;

  return os;
}


/**
 * @brief Specialization of operator << for the 3D case
 */
template<typename T_numtype>
ostream& operator<<(ostream& os, const Array<T_numtype,3>& x)
{
  return out3D_template<T_numtype,T_numtype>(os,x);
}

/**
 * @brief Specialization of operator << for the 4D case
 */
template<typename T_numtype>
ostream& operator<<(ostream& os, const Array<T_numtype,4>& x)
{
  return out4D_template<T_numtype,T_numtype>(os,x);
}



/** 
 * @brief Specialization of operator << for int8_t in 1D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,1>& x);

/** 
 * @brief Specialization of operator << for int8_t in 2D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,2>& x);

/**
 * @brief Specialization of operator << for int8_t in 3D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,3>& x);

/**
 * @brief Specialization of operator << for int8_t in 4D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,4>& x);

/** 
 * @brief Specialization of operator << for uint8_t in 1D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,1>& x);

/** 
 * @brief Specialization of operator << for uint8_t in 2D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,2>& x);

/**
 * @brief Specialization of operator << for uint8_t in 3D
 */
template<>
ostream& operator<<(ostream& os, const Array<uint8_t,3>& x);

/**
 * @brief Specialization of operator << for uint8_t in 4D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,4>& x);




/**
 * @brief Local function for reading a int8/uint8 Blitz++ array,
 * by converting the saved printable type (int16/uint16)
 */
template<typename Tsrc, typename Tdst, int N_rank>
istream& in_template(istream& is, Array<Tdst,N_rank>& x)
{
  TinyVector<int,N_rank> lower_bounds, upper_bounds, extent;
  char sep;

  // Read the range-extent vector: this is separated by 'x's, e.g.
  // (0,2) x (0,4) x (0,5)

  for (int i=0; i < N_rank; ++i) {
    is >> sep;
    BZPRECHECK(!is.bad(), "Premature end of input while scanning Array");
    BZPRECHECK(sep == '(', "Format error while scanning input \
        Array \n -- expected '(' opening Array extents");

    is >> lower_bounds(i); 
    is >> sep; 
    BZPRECHECK(sep == ',', "Format error while scanning input \
        Array \n -- expected ',' between Array extents");
    is >> upper_bounds(i);

    is >> sep; 
    BZPRECHECK(sep == ')', "Format error while scanning input \
        Array \n -- expected ',' closing Array extents");

    if (i != N_rank-1) {
      is >> sep;
      BZPRECHECK(sep == 'x', "Format error while scanning input \
          Array \n (expected 'x' between Array extents)");
    }   
  }

  is >> sep;
  BZPRECHECK(sep == '[', "Format error while scanning input \
      Array \n (expected '[' before beginning of Array data)");

  for (int i=0; i < N_rank; ++i)
    extent(i) = upper_bounds(i) - lower_bounds(i) + 1;
  x.resize(extent);
  x.reindexSelf(lower_bounds);

  _bz_typename Array<Tdst,N_rank>::iterator iter = x.begin();
  _bz_typename Array<Tdst,N_rank>::iterator end = x.end();

  Tsrc temp_val;
  while (iter != end) {
    BZPRECHECK(!is.bad(), "Premature end of input while scanning array");

    is >> temp_val;
    (*iter) = static_cast<Tdst>(temp_val);
    ++iter;
  }

  is >> sep;
  BZPRECHECK(sep == ']', "Format error while scanning input array"
      << endl << " (expected ']' after end of array data)");

  return is;
}


/**
 * @brief Specialization of operator >> for the int8_t case
 */
template<int N_rank>
istream& operator>>(istream& is, Array<int8_t,N_rank>& x)
{
  return in_template<int16_t,int8_t,N_rank>(is, x);
}

/**
 * @brief Specialization of operator >> for the uint8_t case
 */
template<int N_rank>
istream& operator>>(istream& is, Array<uint8_t,N_rank>& x)
{
  return in_template<uint16_t,uint8_t,N_rank>(is, x);
}
#endif
BZ_NAMESPACE_END

#endif /* TORCH_CORE_BLITZ_MISC_H */

