/**
 * @file bob/sp/conv.h
 * @date Thu Feb 3 16:39:25 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based convolution product with zero padding
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_SP_CONV_H
#define BOB_SP_CONV_H

#include <stdexcept>
#include <algorithm>
#include <blitz/array.h>
#include <boost/format.hpp>

#include <bob/core/assert.h>

/**
 * @addtogroup SP sp
 * @brief Signal Processing module API
 */

namespace bob {
/**
 * @ingroup SP
 */
namespace sp {

/**
 * @ingroup SP
 * @brief Enumerations of the possible output size options
 */
namespace Conv {
  typedef enum SizeOption_ {
    Full,
    Same,
    Valid
  } SizeOption;
}

namespace detail {
  template <typename T>
  void convInternal(const blitz::Array<T,1> a, const blitz::Array<T,1> b,
    blitz::Array<T,1> c, const int offset_0, const int offset_1)
  {
    const int M = a.extent(0);
    const int P = c.extent(0);

    int bl_offset = 0;
    int bu_offset = offset_1-1;
    int al_offset = 0;
    for (int i=0; i<P; ++i)
    {
      blitz::Array<T,1> b_s = b(blitz::Range(bu_offset,bl_offset,-1));
      blitz::Array<T,1> a_s =
        a(blitz::Range(al_offset,al_offset+bu_offset-bl_offset));
      c(i) = blitz::sum(a_s * b_s);
      if (i < offset_0) ++bu_offset;
      else ++al_offset;
      if (i >= M-offset_1) ++bl_offset;
    }
  }

  template <typename T>
  void convInternal(const blitz::Array<T,2> A, const blitz::Array<T,2> B,
    blitz::Array<T,2> C, const int offset0_0, const int offset0_1,
    const int offset1_0, const int offset1_1)
  {
    const int M0 = A.extent(0);
    const int M1 = A.extent(1);
    const int P0 = C.extent(0);
    const int P1 = C.extent(1);

    int bl_offset0 = 0;
    int bu_offset0 = offset0_1-1;
    int al_offset0 = 0;

    int bl_offset1;
    int bu_offset1;
    int al_offset1;
    for (int i=0; i<P0; ++i)
    {
      blitz::Range r_b0(bu_offset0,bl_offset0,-1);
      blitz::Range r_a0(al_offset0,al_offset0+bu_offset0-bl_offset0);
      bl_offset1 = 0;
      bu_offset1 = offset1_1-1;
      al_offset1 = 0;

      for (int j=0; j<P1; ++j)
      {
        blitz::Array<T,2> B_s = B(r_b0,blitz::Range(bu_offset1,bl_offset1,-1));
        blitz::Array<T,2> A_s = A(r_a0,blitz::Range(al_offset1,al_offset1+bu_offset1-bl_offset1));
        C(i,j) = blitz::sum(A_s * B_s);
        if (j < offset1_0) ++bu_offset1;
        else ++al_offset1;
        if (j >= M1-offset1_1) ++bl_offset1;
      }
      if (i < offset0_0) ++bu_offset0;
      else ++al_offset0;
      if (i >= M0-offset0_1) ++bl_offset0;
    }
  }

}

/**
 * @ingroup SP
 * @{
 */

/**
 * @brief Gets the required size of the output of a 1D convolution product
 * @param a The size of the first input array
 * @param b The size of the second input array
 * @param size_opt:  * Full: full size (default)
 *                   * Same: same size as the largest between a and b
 *                   * Valid: valid (part without padding)
 * @return The size of the output array
 */
inline size_t getConvOutputSize(const size_t a, const size_t b,
  const Conv::SizeOption size_opt = Conv::Full)
{
  if (a<b) {
    boost::format m("The convolutional kernel has its first dimension larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
    m % a % b;
    throw std::runtime_error(m.str());
  }

  size_t res=0;
  // Size of "A + B - 1"
  if (size_opt == Conv::Full)
    res = std::max(0, (int)(a + b - 1));
  // Same size as A
  else if (size_opt == Conv::Same)
    res = a;
  // Size when not allowing any padding
  else
    res = std::max(0, (int)(a - b + 1));
  return res;
}

/**
 * @brief Gets the required size of the output of the convolution product
 * @param a The first input array a
 * @param b The second input array b
 * @param size_opt:  * Full: full size (default)
 *                   * Same: same size as the largest between a and b
 *                   * Valid: valid (part without padding)
 * @return The size of the output array
 */
template<typename T>
const blitz::TinyVector<int,1> getConvOutputSize(
  const blitz::Array<T,1>& a, const blitz::Array<T,1>& b,
  const Conv::SizeOption size_opt = Conv::Full)
{
  blitz::TinyVector<int,1> size;
  size(0) = getConvOutputSize(a.extent(0), b.extent(0), size_opt);
  return size;
}

/**
 * @brief Gets the required size of the output of the convolution product
 * @param A The first input array A
 * @param B The second input array B
 * @param size_opt:  * Full: full size (default)
 *                   * Same: same size as the largest between A and B
 *                   * Valid: valid (part without padding)
 * @return Size of the output
 */
template<typename T>
const blitz::TinyVector<int,2> getConvOutputSize(
  const blitz::Array<T,2>& A, const blitz::Array<T,2>& B,
  const Conv::SizeOption size_opt = Conv::Full)
{
  if (A.extent(0)<B.extent(0)) {
    boost::format m("The convolutional kernel has the first dimension larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
    m % A.extent(0) % B.extent(0);
    throw std::runtime_error(m.str());
  }
  if (A.extent(1)<B.extent(1)) {
    boost::format m("The convolutional kernel has the second dimension larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
    m % A.extent(1) % B.extent(1);
    throw std::runtime_error(m.str());
  }

  blitz::TinyVector<int,2> size;
  size(0) = 0;
  size(1) = 0;

  if (size_opt == Conv::Full) {
    size(0) = std::max(0, A.extent(0) + B.extent(0) - 1);
    size(1) = std::max(0, A.extent(1) + B.extent(1) - 1);
  }
  else if (size_opt == Conv::Same) {
    size(0) = A.extent(0);
    size(1) = A.extent(1);
  }
  else if (size_opt == Conv::Valid) {
    size(0) = std::max(0, A.extent(0) - B.extent(0) + 1);
    size(1) = std::max(0, A.extent(1) - B.extent(1) + 1);
  }

  return size;
}

/**
 * @brief Gets the required size of the output of the separable convolution product
 *        (Convolution of a X-D signal with a 1D kernel)
 * @param A The first input array A
 * @param b The second input array b
 * @param dim The dimension along which to convolve
 * @param size_opt:  * Full: full size (default)
 *                   * Same: same size as the largest between A and B
 *                   * Valid: valid (part without padding)
 * @return Size of the output
 */
template<typename T, int N>
const blitz::TinyVector<int,N> getConvSepOutputSize(const blitz::Array<T,N>& A,
  const blitz::Array<T,1>& b, const size_t dim,
  const Conv::SizeOption size_opt = Conv::Full)
{
  blitz::TinyVector<int,N> res;
  res = A.shape();
  if ((int)dim<N) {
    if (A.extent(dim)<b.extent(0)) {
      boost::format m("The convolutional kernel has dimension %d larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
      m % dim % A.extent(dim) % b.extent(0);
      throw std::runtime_error(m.str());
    }

    int a_size_d = A.extent(dim);
    int b_size = b.extent(0);
    res((int)dim) = getConvOutputSize(a_size_d, b_size, size_opt);
  }
  else {
    boost::format m("Cannot perform a separable convolution along dimension %d. The maximal dimension index for this array is %d. (Please note that indices starts at 0.");
    m % dim % (N-1);
    throw std::runtime_error(m.str());
  }
  return res;
}



/**
 * @brief 1D convolution of blitz arrays: c=a*b
 * @param a The first input array a
 * @param b The second input array b
 * @param c The output array c=a*b
 * @param size_opt:  * Full: full size (default)
 *                   * Same: same size as the largest between A and B
 *                   * Valid: valid (part without padding)
 * @warning a should be larger than the kernel b
 *    The output c should have the correct size
 */
template <typename T>
void conv(const blitz::Array<T,1> a, const blitz::Array<T,1> b,
  blitz::Array<T,1> c, const Conv::SizeOption size_opt = Conv::Full)
{
  const int N = b.extent(0);

  if (a.extent(0)<b.extent(0)) {
    boost::format m("The convolutional kernel has the first dimension larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
    m % a.extent(0) % b.extent(0);
    throw std::runtime_error(m.str());
  }

  if (size_opt == Conv::Full)
    detail::convInternal(a, b, c, N-1, 1);
  else if (size_opt == Conv::Same)
    detail::convInternal(a, b, c, N/2, (N+1)/2);
  else
    detail::convInternal(a, b, c, 0, N);
}

/**
 * @brief 2D convolution of blitz arrays: C=A*B
 * @param A The first input array A
 * @param B The second input array B
 * @param C The output array C=A*B
 * @param size_opt:  * Full: full size (default)
 *                   * Same: same size as the largest between A and B
 *                   * Valid: valid (part without padding)
 * @warning A should have larger dimensions than the kernel B
 *   The output C should have the correct size
 */
template <typename T>
void conv(const blitz::Array<T,2> A, const blitz::Array<T,2> B,
  blitz::Array<T,2> C, const Conv::SizeOption size_opt = Conv::Full)
{
  const int N0 = B.extent(0);
  const int N1 = B.extent(1);

  if (A.extent(0)<B.extent(0)) {
    boost::format m("The convolutional kernel has the first dimension larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
    m % A.extent(0) % B.extent(0);
    throw std::runtime_error(m.str());
  }
  if (A.extent(1)<B.extent(1)) {
    boost::format m("The convolutional kernel has the second dimension larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
    m % A.extent(1) % B.extent(1);
    throw std::runtime_error(m.str());
  }

  if (size_opt == Conv::Full)
    detail::convInternal(A, B, C, N0-1, 1, N1-1, 1);
  else if (size_opt == Conv::Same)
    detail::convInternal(A, B, C, N0/2, (N0+1)/2, N1/2, (N1+1)/2);
  else
    detail::convInternal(A, B, C, 0, N0, 0, N1);
}

namespace detail {

  template<typename T> void convSep(const blitz::Array<T,2>& A,
    const blitz::Array<T,1>& b, blitz::Array<T,2>& C,
    const Conv::SizeOption size_opt = Conv::Full)
  {
    for (int i=0; i<A.extent(1); ++i)
    {
      const blitz::Array<T,1> Arow = A(blitz::Range::all(), i);
      blitz::Array<T,1> Crow = C(blitz::Range::all(), i);
      conv(Arow, b, Crow, size_opt);
    }
  }

 template<typename T> void convSep(const blitz::Array<T,3>& A,
    const blitz::Array<T,1>& b, blitz::Array<T,3>& C,
    const Conv::SizeOption size_opt = Conv::Full)
  {
    for (int i=0; i<A.extent(1); ++i)
      for (int j=0; j<A.extent(2); ++j)
      {
        const blitz::Array<T,1> Arow = A(blitz::Range::all(), i, j);
        blitz::Array<T,1> Crow = C(blitz::Range::all(), i, j);
        conv(Arow, b, Crow, size_opt);
      }
  }

  template<typename T> void convSep(const blitz::Array<T,4>& A,
    const blitz::Array<T,1>& b, blitz::Array<T,4>& C,
    const Conv::SizeOption size_opt = Conv::Full)
  {
    for (int i=0; i<A.extent(1); ++i)
      for (int j=0; j<A.extent(2); ++j)
        for (int k=0; k<A.extent(3); ++k)
        {
          const blitz::Array<T,1> Arow = A(blitz::Range::all(), i, j, k);
          blitz::Array<T,1> Crow = C(blitz::Range::all(), i, j, k);
          conv(Arow, b, Crow, size_opt);
        }
  }
}

/**
 * @brief Convolution of a X-D signal with a 1D kernel (for separable convolution)
 *        along the specified dimension (C=A*b)
 * @param A The first input array A
 * @param b The second input array b
 * @param C The output array C=A*b along the dimension d (0 or 1)
 * @param dim The dimension along which to convolve
 * @param size_opt:  * Full: full size (default)
 *                   * Same: same size as the largest between A and b
 *                   * Valid: valid (part without padding)
 * @warning A should have larger dimensions than the kernel b
 *   The output C should have the correct size
 */
template<typename T, int N> void convSep(const blitz::Array<T,N>& A,
  const blitz::Array<T,1>& b, blitz::Array<T,N>& C, const size_t dim,
  const Conv::SizeOption size_opt = Conv::Full)
{
  // Gets the expected size for the results
  const blitz::TinyVector<int,N> Csize = getConvSepOutputSize(A, b, dim, size_opt);

  // Checks that C has the correct size and is zero base
  bob::core::array::assertSameShape(C, Csize);
  bob::core::array::assertZeroBase(C);
  // Checks that A and B are zero base
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(b);

  if (dim==0)
  {
    if (A.extent(dim)<b.extent(0)) {
      boost::format m("The convolutional kernel has the first dimension larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
      m % A.extent(0) % b.extent(0);
      throw std::runtime_error(m.str());
    }
    detail::convSep(A, b, C, size_opt);
  }
  else if ((int)dim<N)
  {
    if (A.extent(dim)<b.extent(0)) {
      boost::format m("The convolutional kernel has dimension %d larger than the corresponding one of the array to process (%d > %d). Our convolution code does not allows. You could try to revert the order of the two arrays.");
      m % dim % A.extent(dim) % b.extent(0);
      throw std::runtime_error(m.str());
    }

    // Ugly fix to support old blitz versions without const transpose()
    // method
    const blitz::Array<T,N> Ap =
      (const_cast<blitz::Array<T,N> *>(&A))->transpose(dim,0);
    blitz::Array<T,N> Cp = C.transpose(dim,0);
    detail::convSep(Ap, b, Cp, size_opt);
  }
  else {
    boost::format m("Cannot perform a separable convolution along dimension %d. The maximal dimension index for this array is %d. (Please note that indices starts at 0.");
    m % dim % (N-1);
    throw std::runtime_error(m.str());
  }
}

/**
 * @}
 */
}}

#endif /* BOB_SP_CONV_H */
