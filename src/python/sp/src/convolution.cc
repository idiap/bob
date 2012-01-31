/**
 * @file python/sp/src/convolution.cc
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

#include "sp/convolution.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;

template <typename T, int N> static void inner_convolve_type_dim
(tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c, 
 const enum bob::sp::Convolution::SizeOption d, 
 const enum bob::sp::Convolution::BorderOption e) {
  blitz::Array<T,N> c_ = c.bz<T,N>();
  bob::sp::convolve(a.bz<T,N>(), b.bz<T,N>(), c_, d, e);
}

template <typename T> static void inner_convolve_dim
(size_t nd, tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c,
 const enum bob::sp::Convolution::SizeOption d, 
 const enum bob::sp::Convolution::BorderOption e) {
  switch (nd) {
    case 1: return inner_convolve_type_dim<T,1>(a,b,c,d,e);
    case 2: return inner_convolve_type_dim<T,2>(a,b,c,d,e);
    default: PYTHON_ERROR(TypeError, "non-separable convolution does not support input array with " SIZE_T_FMT " dimensions", nd);
  }
}

static void convolve(tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c, 
 const enum bob::sp::Convolution::SizeOption d=bob::sp::Convolution::Full, 
 const enum bob::sp::Convolution::BorderOption e=bob::sp::Convolution::Zero)
{
  const ca::typeinfo& info = a.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_convolve_dim<bool>(info.nd, a,b,c,d,e);
    case ca::t_int8: 
      return inner_convolve_dim<int8_t>(info.nd, a,b,c,d,e);
    case ca::t_int16: 
      return inner_convolve_dim<int16_t>(info.nd, a,b,c,d,e);
    case ca::t_int32: 
      return inner_convolve_dim<int32_t>(info.nd, a,b,c,d,e);
    case ca::t_int64: 
      return inner_convolve_dim<int64_t>(info.nd, a,b,c,d,e);
    case ca::t_uint8: 
      return inner_convolve_dim<uint8_t>(info.nd, a,b,c,d,e);
    case ca::t_uint16:
      return inner_convolve_dim<uint16_t>(info.nd, a,b,c,d,e);
    case ca::t_uint32: 
      return inner_convolve_dim<uint32_t>(info.nd, a,b,c,d,e);
    case ca::t_uint64: 
      return inner_convolve_dim<uint64_t>(info.nd, a,b,c,d,e);
    case ca::t_float32:
      return inner_convolve_dim<float>(info.nd, a,b,c,d,e);
    case ca::t_float64: 
      return inner_convolve_dim<double>(info.nd, a,b,c,d,e);
    case ca::t_complex64: 
      return inner_convolve_dim<std::complex<float> >(info.nd, a,b,c,d,e);
    case ca::t_complex128: 
      return inner_convolve_dim<std::complex<double> >(info.nd, a,b,c,d,e);
    default: PYTHON_ERROR(TypeError, "non-separable convolution computation does not support with array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N> static object inner_convolve_size_type_dim
(tp::const_ndarray b, tp::const_ndarray c, 
 const enum bob::sp::Convolution::SizeOption d) {
  return object(bob::sp::getConvolveOutputSize(b.bz<T,N>(), c.bz<T,N>(), d));
}

template <typename T> static object inner_convolve_size_dim
(size_t nd, tp::const_ndarray b, tp::const_ndarray c,
 const enum bob::sp::Convolution::SizeOption d) {
  switch (nd) {
    case 1: return inner_convolve_size_type_dim<T,1>(b,c,d);
    case 2: return inner_convolve_size_type_dim<T,2>(b,c,d);
    default: PYTHON_ERROR(TypeError, "non-separable convolution does not support input array with " SIZE_T_FMT " dimensions", nd);
  }
}

static object convolve_size(tp::const_ndarray b, tp::const_ndarray c, 
 const enum bob::sp::Convolution::SizeOption d=bob::sp::Convolution::Full)
{
  const ca::typeinfo& info = b.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_convolve_size_dim<bool>(info.nd, b,c,d);
    case ca::t_int8: 
      return inner_convolve_size_dim<int8_t>(info.nd, b,c,d);
    case ca::t_int16: 
      return inner_convolve_size_dim<int16_t>(info.nd, b,c,d);
    case ca::t_int32: 
      return inner_convolve_size_dim<int32_t>(info.nd, b,c,d);
    case ca::t_int64: 
      return inner_convolve_size_dim<int64_t>(info.nd, b,c,d);
    case ca::t_uint8: 
      return inner_convolve_size_dim<uint8_t>(info.nd, b,c,d);
    case ca::t_uint16:
      return inner_convolve_size_dim<uint16_t>(info.nd, b,c,d);
    case ca::t_uint32: 
      return inner_convolve_size_dim<uint32_t>(info.nd, b,c,d);
    case ca::t_uint64: 
      return inner_convolve_size_dim<uint64_t>(info.nd, b,c,d);
    case ca::t_float32:
      return inner_convolve_size_dim<float>(info.nd, b,c,d);
    case ca::t_float64: 
      return inner_convolve_size_dim<double>(info.nd, b,c,d);
    case ca::t_complex64: 
      return inner_convolve_size_dim<std::complex<float> >(info.nd, b,c,d);
    case ca::t_complex128: 
      return inner_convolve_size_dim<std::complex<double> >(info.nd, b,c,d);
    default: PYTHON_ERROR(TypeError, "non-separable convolution computation does not support with array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N> static void inner_convolve_sep_type_dim
(tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c, int dim,
 const enum bob::sp::Convolution::SizeOption d, 
 const enum bob::sp::Convolution::BorderOption e) {
  blitz::Array<T,N> c_ = c.bz<T,N>();
  bob::sp::convolveSep(a.bz<T,N>(), b.bz<T,1>(), c_, dim, d, e);
}

template <typename T> static void inner_convolve_sep_dim
(size_t nd, tp::const_ndarray a, tp::const_ndarray b, tp::ndarray c, int dim,
 const enum bob::sp::Convolution::SizeOption d, 
 const enum bob::sp::Convolution::BorderOption e) {
  switch (nd) {
    case 2: return inner_convolve_sep_type_dim<T,2>(a,b,c,dim,d,e);
    case 3: return inner_convolve_sep_type_dim<T,3>(a,b,c,dim,d,e);
    case 4: return inner_convolve_sep_type_dim<T,4>(a,b,c,dim,d,e);
    default: PYTHON_ERROR(TypeError, "separable convolution does not support input array with " SIZE_T_FMT " dimensions", nd);
  }
}

static void convolve_sep(tp::const_ndarray a, tp::const_ndarray b,
 tp::ndarray c, int dim,
 const enum bob::sp::Convolution::SizeOption d=bob::sp::Convolution::Full, 
 const enum bob::sp::Convolution::BorderOption e=bob::sp::Convolution::Zero)
{
  const ca::typeinfo& info = a.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_convolve_sep_dim<bool>(info.nd, a,b,c,dim,d,e);
    case ca::t_int8: 
      return inner_convolve_sep_dim<int8_t>(info.nd, a,b,c,dim,d,e);
    case ca::t_int16:
      return inner_convolve_sep_dim<int16_t>(info.nd, a,b,c,dim,d,e);
    case ca::t_int32: 
      return inner_convolve_sep_dim<int32_t>(info.nd, a,b,c,dim,d,e);
    case ca::t_int64: 
      return inner_convolve_sep_dim<int64_t>(info.nd, a,b,c,dim,d,e);
    case ca::t_uint8: 
      return inner_convolve_sep_dim<uint8_t>(info.nd, a,b,c,dim,d,e);
    case ca::t_uint16: 
      return inner_convolve_sep_dim<uint16_t>(info.nd, a,b,c,dim,d,e);
    case ca::t_uint32: 
      return inner_convolve_sep_dim<uint32_t>(info.nd, a,b,c,dim,d,e);
    case ca::t_uint64: 
      return inner_convolve_sep_dim<uint64_t>(info.nd, a,b,c,dim,d,e);
    case ca::t_float32:
      return inner_convolve_sep_dim<float>(info.nd, a,b,c,dim,d,e);
    case ca::t_float64: 
      return inner_convolve_sep_dim<double>(info.nd, a,b,c,dim,d,e);
    case ca::t_complex64:
      return inner_convolve_sep_dim<std::complex<float> >(info.nd, a,b,c,dim,d,e);
    case ca::t_complex128: 
      return inner_convolve_sep_dim<std::complex<double> >(info.nd, a,b,c,dim,d,e);
    default: PYTHON_ERROR(TypeError, "separable convolution computation does not support input array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N> static object inner_convolve_sep_size_type_dim
(tp::const_ndarray b, tp::const_ndarray c, int dim,
 const enum bob::sp::Convolution::SizeOption d) {
  return object(bob::sp::getConvolveSepOutputSize(b.bz<T,N>(), c.bz<T,1>(), 
        dim, d));
}

template <typename T> static object inner_convolve_sep_size_dim
(size_t nd, tp::const_ndarray b, tp::const_ndarray c, int dim,
 const enum bob::sp::Convolution::SizeOption d) {
  switch (nd) {
    case 2: return inner_convolve_sep_size_type_dim<T,2>(b,c,dim,d);
    case 3: return inner_convolve_sep_size_type_dim<T,3>(b,c,dim,d);
    case 4: return inner_convolve_sep_size_type_dim<T,4>(b,c,dim,d);
    default: PYTHON_ERROR(TypeError, "separable convolution does not support input array with " SIZE_T_FMT " dimensions", nd);
  }
}

static object convolve_sep_size(tp::const_ndarray b, tp::const_ndarray c,
 int dim, const enum bob::sp::Convolution::SizeOption d=bob::sp::Convolution::Full)
{
  const ca::typeinfo& info = b.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_convolve_sep_size_dim<bool>(info.nd, b,c,dim,d);
    case ca::t_int8: 
      return inner_convolve_sep_size_dim<int8_t>(info.nd, b,c,dim,d);
    case ca::t_int16: 
      return inner_convolve_sep_size_dim<int16_t>(info.nd, b,c,dim,d);
    case ca::t_int32: 
      return inner_convolve_sep_size_dim<int32_t>(info.nd, b,c,dim,d);
    case ca::t_int64: 
      return inner_convolve_sep_size_dim<int64_t>(info.nd, b,c,dim,d);
    case ca::t_uint8: 
      return inner_convolve_sep_size_dim<uint8_t>(info.nd, b,c,dim,d);
    case ca::t_uint16:
      return inner_convolve_sep_size_dim<uint16_t>(info.nd, b,c,dim,d);
    case ca::t_uint32: 
      return inner_convolve_sep_size_dim<uint32_t>(info.nd, b,c,dim,d);
    case ca::t_uint64: 
      return inner_convolve_sep_size_dim<uint64_t>(info.nd, b,c,dim,d);
    case ca::t_float32:
      return inner_convolve_sep_size_dim<float>(info.nd, b,c,dim,d);
    case ca::t_float64: 
      return inner_convolve_sep_size_dim<double>(info.nd, b,c,dim,d);
    case ca::t_complex64: 
      return inner_convolve_sep_size_dim<std::complex<float> >(info.nd, b,c,dim,d);
    case ca::t_complex128: 
      return inner_convolve_sep_size_dim<std::complex<double> >(info.nd, b,c,dim,d);
    default: PYTHON_ERROR(TypeError, "separable convolution computation does not support with array with type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(convolve_overloads, convolve, 3, 5)
BOOST_PYTHON_FUNCTION_OVERLOADS(convolve_size_overloads, convolve_size, 2, 3)
BOOST_PYTHON_FUNCTION_OVERLOADS(convolve_sep_overloads, convolve_sep, 4, 6)
BOOST_PYTHON_FUNCTION_OVERLOADS(convolve_sep_size_overloads, convolve_sep_size, 3, 4)

void bind_sp_convolution()
{
  enum_<bob::sp::Convolution::SizeOption>("ConvolutionSize")
    .value("Full", bob::sp::Convolution::Full)
    .value("Same", bob::sp::Convolution::Same)
    .value("Valid", bob::sp::Convolution::Valid)
    ;
 
  enum_<bob::sp::Convolution::BorderOption>("ConvolutionBorder")
    .value("Zero", bob::sp::Convolution::Zero)
    .value("NearestNeighbour", bob::sp::Convolution::NearestNeighbour)
    .value("Circular", bob::sp::Convolution::Circular)
    .value("Mirror", bob::sp::Convolution::Mirror)
    ;
 
  def("convolve", &convolve, convolve_overloads((arg("b"), arg("c"), arg("a"), arg("size_opt")="Full", arg("border_opt")="Zero"), "Compute the convolution product of two blitz arrays using zero padding and return the results as a blitz array. The option field allows to give information about the size of the output (FULL, SAME, VALID)"));

  def("getConvolveOutputSize", &convolve_size, convolve_size_overloads((arg("b"), arg("c"), arg("size_opt")="full"), "gets the required size of the result of a 1 or 2D convolution product"));

  def("convolveSep", &convolve_sep, convolve_sep_overloads((arg("b"), arg("c"), arg("a"), arg("dim"), arg("size_opt")="Full", arg("border_opt")="Zero"), "Computes the convolution product of a 2, 3 or 4D blitz array with a 1D one, along the given dimension. (useful for separable convolution)"));

  def("getConvolveSepOutputSize", &convolve_sep_size, convolve_sep_size_overloads((arg("b"), arg("c"), arg("dim"), arg("size_opt")="Full"), "Gets the required size of the result of a 2, 3 or 4D separable convolution product"));
}
