/**
 * @file python/sp/src/conv.cc
 * @date Fri Jan 28 13:27:58 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds convolution product to python
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "sp/conv.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;

template <typename T, int N> static void inner_conv_type_dim
(tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c, 
 const enum bob::sp::Conv::SizeOption d) { 
  blitz::Array<T,N> c_ = c.bz<T,N>();
  bob::sp::conv(a.bz<T,N>(), b.bz<T,N>(), c_, d);
}

template <typename T> static void inner_conv_dim
(size_t nd, tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c,
 const enum bob::sp::Conv::SizeOption d) { 
  switch (nd) {
    case 1: return inner_conv_type_dim<T,1>(a,b,c,d);
    case 2: return inner_conv_type_dim<T,2>(a,b,c,d);
    default: PYTHON_ERROR(TypeError, "non-separable convolution does not support input array with " SIZE_T_FMT " dimensions", nd);
  }
}

static void conv(tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c, 
 const enum bob::sp::Conv::SizeOption d=bob::sp::Conv::Full) 
{
  const ca::typeinfo& info = a.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_conv_dim<bool>(info.nd, a,b,c,d);
    case ca::t_int8: 
      return inner_conv_dim<int8_t>(info.nd, a,b,c,d);
    case ca::t_int16: 
      return inner_conv_dim<int16_t>(info.nd, a,b,c,d);
    case ca::t_int32: 
      return inner_conv_dim<int32_t>(info.nd, a,b,c,d);
    case ca::t_int64: 
      return inner_conv_dim<int64_t>(info.nd, a,b,c,d);
    case ca::t_uint8: 
      return inner_conv_dim<uint8_t>(info.nd, a,b,c,d);
    case ca::t_uint16:
      return inner_conv_dim<uint16_t>(info.nd, a,b,c,d);
    case ca::t_uint32: 
      return inner_conv_dim<uint32_t>(info.nd, a,b,c,d);
    case ca::t_uint64: 
      return inner_conv_dim<uint64_t>(info.nd, a,b,c,d);
    case ca::t_float32:
      return inner_conv_dim<float>(info.nd, a,b,c,d);
    case ca::t_float64: 
      return inner_conv_dim<double>(info.nd, a,b,c,d);
    case ca::t_complex64: 
      return inner_conv_dim<std::complex<float> >(info.nd, a,b,c,d);
    case ca::t_complex128: 
      return inner_conv_dim<std::complex<double> >(info.nd, a,b,c,d);
    default: PYTHON_ERROR(TypeError, "non-separable convolution computation does not support with array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N> static object inner_conv_size_type_dim
(tp::const_ndarray b, tp::const_ndarray c, 
 const enum bob::sp::Conv::SizeOption d) {
  return object(bob::sp::getConvOutputSize(b.bz<T,N>(), c.bz<T,N>(), d));
}

template <typename T> static object inner_conv_size_dim
(size_t nd, tp::const_ndarray b, tp::const_ndarray c,
 const enum bob::sp::Conv::SizeOption d) {
  switch (nd) {
    case 1: return inner_conv_size_type_dim<T,1>(b,c,d);
    case 2: return inner_conv_size_type_dim<T,2>(b,c,d);
    default: PYTHON_ERROR(TypeError, "non-separable convolution does not support input array with " SIZE_T_FMT " dimensions", nd);
  }
}

static object conv_size(tp::const_ndarray b, tp::const_ndarray c, 
 const enum bob::sp::Conv::SizeOption d=bob::sp::Conv::Full)
{
  const ca::typeinfo& info = b.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_conv_size_dim<bool>(info.nd, b,c,d);
    case ca::t_int8: 
      return inner_conv_size_dim<int8_t>(info.nd, b,c,d);
    case ca::t_int16: 
      return inner_conv_size_dim<int16_t>(info.nd, b,c,d);
    case ca::t_int32: 
      return inner_conv_size_dim<int32_t>(info.nd, b,c,d);
    case ca::t_int64: 
      return inner_conv_size_dim<int64_t>(info.nd, b,c,d);
    case ca::t_uint8: 
      return inner_conv_size_dim<uint8_t>(info.nd, b,c,d);
    case ca::t_uint16:
      return inner_conv_size_dim<uint16_t>(info.nd, b,c,d);
    case ca::t_uint32: 
      return inner_conv_size_dim<uint32_t>(info.nd, b,c,d);
    case ca::t_uint64: 
      return inner_conv_size_dim<uint64_t>(info.nd, b,c,d);
    case ca::t_float32:
      return inner_conv_size_dim<float>(info.nd, b,c,d);
    case ca::t_float64: 
      return inner_conv_size_dim<double>(info.nd, b,c,d);
    case ca::t_complex64: 
      return inner_conv_size_dim<std::complex<float> >(info.nd, b,c,d);
    case ca::t_complex128: 
      return inner_conv_size_dim<std::complex<double> >(info.nd, b,c,d);
    default: PYTHON_ERROR(TypeError, "non-separable convolution computation does not support with array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N> static void inner_conv_sep_type_dim
(tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c, int dim,
 const enum bob::sp::Conv::SizeOption d) {
  blitz::Array<T,N> c_ = c.bz<T,N>();
  bob::sp::convSep(a.bz<T,N>(), b.bz<T,1>(), c_, dim, d);
}

template <typename T> static void inner_conv_sep_dim
(size_t nd, tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c, int dim,
 const enum bob::sp::Conv::SizeOption d) { 
  switch (nd) {
    case 2: return inner_conv_sep_type_dim<T,2>(a,b,c,dim,d);
    case 3: return inner_conv_sep_type_dim<T,3>(a,b,c,dim,d);
    case 4: return inner_conv_sep_type_dim<T,4>(a,b,c,dim,d);
    default: PYTHON_ERROR(TypeError, "separable convolution does not support input array with " SIZE_T_FMT " dimensions", nd);
  }
}

static void conv_sep(tp::const_ndarray a, tp::const_ndarray b,
 tp::ndarray c, int dim,
 const enum bob::sp::Conv::SizeOption d=bob::sp::Conv::Full) 
{
  const ca::typeinfo& info = a.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_conv_sep_dim<bool>(info.nd, a,b,c,dim,d);
    case ca::t_int8: 
      return inner_conv_sep_dim<int8_t>(info.nd, a,b,c,dim,d);
    case ca::t_int16:
      return inner_conv_sep_dim<int16_t>(info.nd, a,b,c,dim,d);
    case ca::t_int32: 
      return inner_conv_sep_dim<int32_t>(info.nd, a,b,c,dim,d);
    case ca::t_int64: 
      return inner_conv_sep_dim<int64_t>(info.nd, a,b,c,dim,d);
    case ca::t_uint8: 
      return inner_conv_sep_dim<uint8_t>(info.nd, a,b,c,dim,d);
    case ca::t_uint16: 
      return inner_conv_sep_dim<uint16_t>(info.nd, a,b,c,dim,d);
    case ca::t_uint32: 
      return inner_conv_sep_dim<uint32_t>(info.nd, a,b,c,dim,d);
    case ca::t_uint64: 
      return inner_conv_sep_dim<uint64_t>(info.nd, a,b,c,dim,d);
    case ca::t_float32:
      return inner_conv_sep_dim<float>(info.nd, a,b,c,dim,d);
    case ca::t_float64: 
      return inner_conv_sep_dim<double>(info.nd, a,b,c,dim,d);
    case ca::t_complex64:
      return inner_conv_sep_dim<std::complex<float> >(info.nd, a,b,c,dim,d);
    case ca::t_complex128: 
      return inner_conv_sep_dim<std::complex<double> >(info.nd, a,b,c,dim,d);
    default: PYTHON_ERROR(TypeError, "separable convolution computation does not support input array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N> static object inner_conv_sep_size_type_dim
(tp::const_ndarray b, tp::const_ndarray c, int dim,
 const enum bob::sp::Conv::SizeOption d) {
  return object(bob::sp::getConvSepOutputSize(b.bz<T,N>(), c.bz<T,1>(), 
        dim, d));
}

template <typename T> static object inner_conv_sep_size_dim
(size_t nd, tp::const_ndarray b, tp::const_ndarray c, int dim,
 const enum bob::sp::Conv::SizeOption d) {
  switch (nd) {
    case 2: return inner_conv_sep_size_type_dim<T,2>(b,c,dim,d);
    case 3: return inner_conv_sep_size_type_dim<T,3>(b,c,dim,d);
    case 4: return inner_conv_sep_size_type_dim<T,4>(b,c,dim,d);
    default: PYTHON_ERROR(TypeError, "separable convolution does not support input array with " SIZE_T_FMT " dimensions", nd);
  }
}

static object conv_sep_size(tp::const_ndarray b, tp::const_ndarray c,
 int dim, const enum bob::sp::Conv::SizeOption d=bob::sp::Conv::Full)
{
  const ca::typeinfo& info = b.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_conv_sep_size_dim<bool>(info.nd, b,c,dim,d);
    case ca::t_int8: 
      return inner_conv_sep_size_dim<int8_t>(info.nd, b,c,dim,d);
    case ca::t_int16: 
      return inner_conv_sep_size_dim<int16_t>(info.nd, b,c,dim,d);
    case ca::t_int32: 
      return inner_conv_sep_size_dim<int32_t>(info.nd, b,c,dim,d);
    case ca::t_int64: 
      return inner_conv_sep_size_dim<int64_t>(info.nd, b,c,dim,d);
    case ca::t_uint8: 
      return inner_conv_sep_size_dim<uint8_t>(info.nd, b,c,dim,d);
    case ca::t_uint16:
      return inner_conv_sep_size_dim<uint16_t>(info.nd, b,c,dim,d);
    case ca::t_uint32: 
      return inner_conv_sep_size_dim<uint32_t>(info.nd, b,c,dim,d);
    case ca::t_uint64: 
      return inner_conv_sep_size_dim<uint64_t>(info.nd, b,c,dim,d);
    case ca::t_float32:
      return inner_conv_sep_size_dim<float>(info.nd, b,c,dim,d);
    case ca::t_float64: 
      return inner_conv_sep_size_dim<double>(info.nd, b,c,dim,d);
    case ca::t_complex64: 
      return inner_conv_sep_size_dim<std::complex<float> >(info.nd, b,c,dim,d);
    case ca::t_complex128: 
      return inner_conv_sep_size_dim<std::complex<double> >(info.nd, b,c,dim,d);
    default: PYTHON_ERROR(TypeError, "separable convolution computation does not support with array with type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(conv_overloads, conv, 3, 4)
BOOST_PYTHON_FUNCTION_OVERLOADS(conv_size_overloads, conv_size, 2, 3)
BOOST_PYTHON_FUNCTION_OVERLOADS(conv_sep_overloads, conv_sep, 4, 5)
BOOST_PYTHON_FUNCTION_OVERLOADS(conv_sep_size_overloads, conv_sep_size, 3, 4)

void bind_sp_conv()
{
  enum_<bob::sp::Conv::SizeOption>("ConvSize")
    .value("Full", bob::sp::Conv::Full)
    .value("Same", bob::sp::Conv::Same)
    .value("Valid", bob::sp::Conv::Valid)
    ;
 
  def("conv", &conv, conv_overloads((arg("b"), arg("c"), arg("a"), arg("size_opt")="Full"), "Compute the convolution product of two blitz arrays using zero padding and return the results as a blitz array. The option field allows to give information about the size of the output (FULL, SAME, VALID)"));

  def("getConvOutputSize", &conv_size, conv_size_overloads((arg("b"), arg("c"), arg("size_opt")="full"), "gets the required size of the result of a 1 or 2D convolution product"));

  def("convSep", &conv_sep, conv_sep_overloads((arg("b"), arg("c"), arg("a"), arg("dim"), arg("size_opt")="Full"), "Computes the convolution product of a 2, 3 or 4D blitz array with a 1D one, along the given dimension. (useful for separable convolution)"));

  def("getConvSepOutputSize", &conv_sep_size, conv_sep_size_overloads((arg("b"), arg("c"), arg("dim"), arg("size_opt")="Full"), "Gets the required size of the result of a 2, 3 or 4D separable convolution product"));
}
