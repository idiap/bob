/**
 * @file src/core/core/blitz_misc.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file contains miscellaneous additions for blitz++ arrays
 *
 * http://www.oonumerics.org/blitz/
 */

#ifndef TORCH_CORE_BLITZ_MISC_H
#define TORCH_CORE_BLITZ_MISC_H

#include <blitz/array.h>
#include "core/logging.h"

// TODO: Is it a reasonable choice to keep the following together?

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    namespace array {

      /**
       * @brief Enumeration of the supported type for multidimensional arrays
       */
      typedef enum ArrayType { t_unknown, t_bool,
        t_int8, t_int16, t_int32, t_int64,
        t_uint8, t_uint16, t_uint32, t_uint64,
        t_float32, t_float64, t_float128,
        t_complex64, t_complex128, t_complex256 } ArrayType;

      /**
       * @brief Maximum number of supported dimensions for multidimensional 
       * arrays.
       */
      const size_t N_MAX_DIMENSIONS_ARRAY = 4;

    }

  }
/**
 * @}
 */
}



BZ_NAMESPACE(blitz)
//TODO
#ifdef TORCH_NEEDS_REVISION
/**
 * @brief Local function for outputting a 1D int8/uint8 Blitz++ array,
 * by converting it to a printable type (int16/uint16)
 */
template <typename Tsrc, typename Tdst>
ostream& out1D_template(ostream& os, const Array<Tsrc,1>& x)
{
  os << x.extent(firstRank) << endl;
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
  os << x.rows() << " x " << x.columns() << endl;
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
  int N_rank = 3;
  for (int i=0; i < N_rank; ++i) {
    os << x.extent(i);
    if (i != N_rank - 1)
      os << " x ";
  }

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
  int N_rank = 4;
  for (int i=0; i < N_rank; ++i) {
    os << x.extent(i);
    if (i != N_rank - 1)
      os << " x ";
  }

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
    TinyVector<int,N_rank> extent;
    char sep;
 
    // Read the extent vector: this is separated by 'x's, e.g.
    // 3 x 4 x 5

    for (int i=0; i < N_rank; ++i)
    {
        is >> extent(i);

        BZPRECHECK(!is.bad(), "Premature end of input while scanning array");

        if (i != N_rank - 1)
        {
            is >> sep;
            BZPRECHECK(sep == 'x', "Format error while scanning input array"
                << endl << " (expected 'x' between array extents)");
        }
    }

    is >> sep;
    BZPRECHECK(sep == '[', "Format error while scanning input array"
        << endl << " (expected '[' before beginning of array data)");

    x.resize(extent);

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

