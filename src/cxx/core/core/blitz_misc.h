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
#include "core/StaticComplexCast.h"

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
 */
template <typename T> 
Array<T,1> copySafedata( const Array<T,1>& src)
{
  int n_0 = src.extent(0);
  Array<T,1> dst( n_0 );
  for( int i=0; i<n_0; ++i )
    dst(i) = src(i+src.lbound(0));
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
  int n_0 = src.extent(0);
  int n_1 = src.extent(1);
  Array<T,2> dst( n_0, n_1 );
  for( int i=0; i<n_0; ++i )
    for( int j=0; j<n_1; ++j )
      dst(i,j) = src(i+src.lbound(0),j+src.lbound(1));
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
  int n_0 = src.extent(0);
  int n_1 = src.extent(1);
  int n_2 = src.extent(2);
  Array<T,3> dst( n_0, n_1, n_2 );
  for( int i=0; i<n_0; ++i )
    for( int j=0; j<n_1; ++j )
      for( int k=0; k<n_2; ++k )
        dst(i,j,k) = src(i+src.lbound(0),j+src.lbound(1),k+src.lbound(2));
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
  int n_0 = src.extent(0);
  int n_1 = src.extent(1);
  int n_2 = src.extent(2);
  int n_3 = src.extent(3);
  Array<T,4> dst( n_0, n_1, n_2, n_3 );
  for( int i=0; i<n_0; ++i )
    for( int j=0; j<n_1; ++j )
      for( int k=0; k<n_2; ++k )
        for( int l=0; l<n_3; ++l )
          dst(i,j,k,l) = src(i+src.lbound(0),j+src.lbound(1),k+src.lbound(2),
            l+src.lbound(3));
  return dst;
}


/**
 * @brief Casts a blitz array allowing std::complex types.
 */
template<typename T, typename U, int D> 
void complex_cast(const Array<U,D> in, Array<T,D> out) {
  out = cast<T>(in);
}

template<typename T, typename U> 
void complex_cast(const Array<std::complex<U>,1> in, Array<T,1> out) {
  for( int i=0; i<in.extent(0); ++i)
    Torch::core::static_complex_cast<T>( in(i+in.lbound(0)), out(i));
}

template<typename T, typename U> 
void complex_cast(const Array<std::complex<U>,2> in, Array<T,2> out) {
  for( int i=0; i<in.extent(0); ++i)
    for( int j=0; j<in.extent(1); ++j)
      Torch::core::static_complex_cast<T>( in(i+in.lbound(0),j+in.lbound(1)), out(i,j));
}

template<typename T, typename U> 
void complex_cast(const Array<std::complex<U>,3> in, Array<T,3> out) {
  for( int i=0; i<in.extent(0); ++i)
    for( int j=0; j<in.extent(1); ++j)
      for( int k=0; k<in.extent(2); ++k)
        Torch::core::static_complex_cast<T>( in(i+in.lbound(0),j+in.lbound(1),k+in.lbound(2)), out(i,j,k));
}

template<typename T, typename U> 
void complex_cast(const Array<std::complex<U>,4> in, Array<T,4> out) {
  for( int i=0; i<in.extent(0); ++i)
    for( int j=0; j<in.extent(1); ++j)
      for( int k=0; k<in.extent(2); ++k)
        for( int l=0; l<in.extent(3); ++l)
          Torch::core::static_complex_cast<T>( in(i+in.lbound(0),j+in.lbound(1),k+in.lbound(2),l+in.lbound(3)), out(i,j,k,l));
}


template<typename T, typename U, int D> 
void complex_cast(const Array<std::complex<U>,D> in, Array<std::complex<T>,D> out) {
  out = cast<std::complex<T> >(in);
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

