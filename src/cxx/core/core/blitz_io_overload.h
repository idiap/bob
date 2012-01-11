/**
 * @file cxx/core/core/blitz_io_overload.h
 * @date Mon Apr 11 10:29:29 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file overloads the Input/Output stream operations on blitz++
 *   multidimensional arrays.
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

#ifndef BOB_CORE_BLITZ_IO_OVERLOAD_H
#define BOB_CORE_BLITZ_IO_OVERLOAD_H

#include <cstdlib>
#include <blitz/array.h>

BZ_NAMESPACE(blitz)

//TODO
#ifdef BOB_NEEDS_REVISION
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

#endif /* BOB_CORE_BLITZ_IO_OVERLOAD_H */
