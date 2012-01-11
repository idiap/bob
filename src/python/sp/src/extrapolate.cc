/**
 * @file python/sp/src/extrapolate.cc
 * @date Tue Sep 27 23:26:46 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds extrapolation to python
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

#include "core/python/ndarray.h"
#include "sp/extrapolate.h"

using namespace boost::python;
namespace tp = bob::python;
namespace sp = bob::sp;
namespace ca = bob::core::array;

template <typename T, int N>
static void inner_extrapolateConstant_dim_size(tp::const_ndarray a,
    tp::ndarray b, object c) {
  blitz::Array<T,N> b_ = b.bz<T,N>();
  sp::extrapolateConstant<T>(a.bz<T,N>(), b_, extract<T>(c));
}

template <typename T>
static void inner_extrapolateConstant_dim(size_t nd, tp::const_ndarray a,
    tp::ndarray b, object c) {
  switch (nd) {
    case 1: return inner_extrapolateConstant_dim_size<T,1>(a,b,c);
    case 2: return inner_extrapolateConstant_dim_size<T,2>(a,b,c);
    default: PYTHON_ERROR(TypeError, "constant extrapolation not supported for array with " SIZE_T_FMT " dimensions", nd);
  }
}

static void extrapolateConstant(tp::const_ndarray a, tp::ndarray b, object c) {
  const ca::typeinfo& info = a.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_extrapolateConstant_dim<bool>(info.nd, a,b,c);
    case ca::t_int8: 
      return inner_extrapolateConstant_dim<int8_t>(info.nd, a,b,c);
    case ca::t_int16: 
      return inner_extrapolateConstant_dim<int16_t>(info.nd, a,b,c);
    case ca::t_int32: 
      return inner_extrapolateConstant_dim<int32_t>(info.nd, a,b,c);
    case ca::t_int64: 
      return inner_extrapolateConstant_dim<int64_t>(info.nd, a,b,c);
    case ca::t_uint8: 
      return inner_extrapolateConstant_dim<uint8_t>(info.nd, a,b,c);
    case ca::t_uint16:
      return inner_extrapolateConstant_dim<uint16_t>(info.nd, a,b,c);
    case ca::t_uint32: 
      return inner_extrapolateConstant_dim<uint32_t>(info.nd, a,b,c);
    case ca::t_uint64: 
      return inner_extrapolateConstant_dim<uint64_t>(info.nd, a,b,c);
    case ca::t_float32:
      return inner_extrapolateConstant_dim<float>(info.nd, a,b,c);
    case ca::t_float64: 
      return inner_extrapolateConstant_dim<double>(info.nd, a,b,c);
    case ca::t_complex64: 
      return inner_extrapolateConstant_dim<std::complex<float> >(info.nd, a,b,c);
    case ca::t_complex128: 
      return inner_extrapolateConstant_dim<std::complex<double> >(info.nd, a,b,c);
    default: PYTHON_ERROR(TypeError, "constant extrapolation not supported for array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_extrapolateZero_dim_size(tp::const_ndarray a,
    tp::ndarray b) {
  blitz::Array<T,N> b_ = b.bz<T,N>();
  sp::extrapolateZero<T>(a.bz<T,N>(), b_);
}

template <typename T>
static void inner_extrapolateZero_dim(size_t nd, tp::const_ndarray a,
    tp::ndarray b) {
  switch (nd) {
    case 1: return inner_extrapolateZero_dim_size<T,1>(a,b);
    case 2: return inner_extrapolateZero_dim_size<T,2>(a,b);
    default: PYTHON_ERROR(TypeError, "zero extrapolation not supported for array with " SIZE_T_FMT " dimensions", nd);
  }
}

static void extrapolateZero(tp::const_ndarray a, tp::ndarray b) {
  const ca::typeinfo& info = a.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_extrapolateZero_dim<bool>(info.nd, a,b);
    case ca::t_int8: 
      return inner_extrapolateZero_dim<int8_t>(info.nd, a,b);
    case ca::t_int16: 
      return inner_extrapolateZero_dim<int16_t>(info.nd, a,b);
    case ca::t_int32: 
      return inner_extrapolateZero_dim<int32_t>(info.nd, a,b);
    case ca::t_int64: 
      return inner_extrapolateZero_dim<int64_t>(info.nd, a,b);
    case ca::t_uint8: 
      return inner_extrapolateZero_dim<uint8_t>(info.nd, a,b);
    case ca::t_uint16:
      return inner_extrapolateZero_dim<uint16_t>(info.nd, a,b);
    case ca::t_uint32: 
      return inner_extrapolateZero_dim<uint32_t>(info.nd, a,b);
    case ca::t_uint64: 
      return inner_extrapolateZero_dim<uint64_t>(info.nd, a,b);
    case ca::t_float32:
      return inner_extrapolateZero_dim<float>(info.nd, a,b);
    case ca::t_float64: 
      return inner_extrapolateZero_dim<double>(info.nd, a,b);
    case ca::t_complex64: 
      return inner_extrapolateZero_dim<std::complex<float> >(info.nd, a,b);
    case ca::t_complex128: 
      return inner_extrapolateZero_dim<std::complex<double> >(info.nd, a,b);
    default: PYTHON_ERROR(TypeError, "zero extrapolation not supported for array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_extrapolateNearest_dim_size(tp::const_ndarray a,
    tp::ndarray b) {
  blitz::Array<T,N> b_ = b.bz<T,N>();
  sp::extrapolateNearest<T>(a.bz<T,N>(), b_);
}

template <typename T>
static void inner_extrapolateNearest_dim(size_t nd, tp::const_ndarray a,
    tp::ndarray b) {
  switch (nd) {
    case 1: return inner_extrapolateNearest_dim_size<T,1>(a,b);
    case 2: return inner_extrapolateNearest_dim_size<T,2>(a,b);
    default: PYTHON_ERROR(TypeError, "nearest extrapolation not supported for array with " SIZE_T_FMT " dimensions", nd);
  }
}

static void extrapolateNearest(tp::const_ndarray a, tp::ndarray b) {
  const ca::typeinfo& info = a.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_extrapolateNearest_dim<bool>(info.nd, a,b);
    case ca::t_int8: 
      return inner_extrapolateNearest_dim<int8_t>(info.nd, a,b);
    case ca::t_int16: 
      return inner_extrapolateNearest_dim<int16_t>(info.nd, a,b);
    case ca::t_int32: 
      return inner_extrapolateNearest_dim<int32_t>(info.nd, a,b);
    case ca::t_int64: 
      return inner_extrapolateNearest_dim<int64_t>(info.nd, a,b);
    case ca::t_uint8: 
      return inner_extrapolateNearest_dim<uint8_t>(info.nd, a,b);
    case ca::t_uint16:
      return inner_extrapolateNearest_dim<uint16_t>(info.nd, a,b);
    case ca::t_uint32: 
      return inner_extrapolateNearest_dim<uint32_t>(info.nd, a,b);
    case ca::t_uint64: 
      return inner_extrapolateNearest_dim<uint64_t>(info.nd, a,b);
    case ca::t_float32:
      return inner_extrapolateNearest_dim<float>(info.nd, a,b);
    case ca::t_float64: 
      return inner_extrapolateNearest_dim<double>(info.nd, a,b);
    case ca::t_complex64: 
      return inner_extrapolateNearest_dim<std::complex<float> >(info.nd, a,b);
    case ca::t_complex128: 
      return inner_extrapolateNearest_dim<std::complex<double> >(info.nd, a,b);
    default: PYTHON_ERROR(TypeError, "nearest extrapolation not supported for array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_extrapolateCircular_dim_size(tp::const_ndarray a,
    tp::ndarray b) {
  blitz::Array<T,N> b_ = b.bz<T,N>();
  sp::extrapolateCircular<T>(a.bz<T,N>(), b_);
}

template <typename T>
static void inner_extrapolateCircular_dim(size_t nd, tp::const_ndarray a,
    tp::ndarray b) {
  switch (nd) {
    case 1: return inner_extrapolateCircular_dim_size<T,1>(a,b);
    case 2: return inner_extrapolateCircular_dim_size<T,2>(a,b);
    default: PYTHON_ERROR(TypeError, "circular extrapolation not supported for array with " SIZE_T_FMT " dimensions", nd);
  }
}

static void extrapolateCircular(tp::const_ndarray a, tp::ndarray b) {
  const ca::typeinfo& info = a.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_extrapolateCircular_dim<bool>(info.nd, a,b);
    case ca::t_int8: 
      return inner_extrapolateCircular_dim<int8_t>(info.nd, a,b);
    case ca::t_int16: 
      return inner_extrapolateCircular_dim<int16_t>(info.nd, a,b);
    case ca::t_int32: 
      return inner_extrapolateCircular_dim<int32_t>(info.nd, a,b);
    case ca::t_int64: 
      return inner_extrapolateCircular_dim<int64_t>(info.nd, a,b);
    case ca::t_uint8: 
      return inner_extrapolateCircular_dim<uint8_t>(info.nd, a,b);
    case ca::t_uint16:
      return inner_extrapolateCircular_dim<uint16_t>(info.nd, a,b);
    case ca::t_uint32: 
      return inner_extrapolateCircular_dim<uint32_t>(info.nd, a,b);
    case ca::t_uint64: 
      return inner_extrapolateCircular_dim<uint64_t>(info.nd, a,b);
    case ca::t_float32:
      return inner_extrapolateCircular_dim<float>(info.nd, a,b);
    case ca::t_float64: 
      return inner_extrapolateCircular_dim<double>(info.nd, a,b);
    case ca::t_complex64: 
      return inner_extrapolateCircular_dim<std::complex<float> >(info.nd, a,b);
    case ca::t_complex128: 
      return inner_extrapolateCircular_dim<std::complex<double> >(info.nd, a,b);
    default: PYTHON_ERROR(TypeError, "circular extrapolation not supported for array with type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_extrapolateMirror_dim_size(tp::const_ndarray a,
    tp::ndarray b) {
  blitz::Array<T,N> b_ = b.bz<T,N>();
  sp::extrapolateMirror<T>(a.bz<T,N>(), b_);
}

template <typename T>
static void inner_extrapolateMirror_dim(size_t nd, tp::const_ndarray a,
    tp::ndarray b) {
  switch (nd) {
    case 1: return inner_extrapolateMirror_dim_size<T,1>(a,b);
    case 2: return inner_extrapolateMirror_dim_size<T,2>(a,b);
    default: PYTHON_ERROR(TypeError, "mirror extrapolation not supported for array with " SIZE_T_FMT " dimensions", nd);
  }
}

static void extrapolateMirror(tp::const_ndarray a, tp::ndarray b) {
  const ca::typeinfo& info = a.type();
  switch (info.dtype) {
    case ca::t_bool: 
      return inner_extrapolateMirror_dim<bool>(info.nd, a,b);
    case ca::t_int8: 
      return inner_extrapolateMirror_dim<int8_t>(info.nd, a,b);
    case ca::t_int16: 
      return inner_extrapolateMirror_dim<int16_t>(info.nd, a,b);
    case ca::t_int32: 
      return inner_extrapolateMirror_dim<int32_t>(info.nd, a,b);
    case ca::t_int64: 
      return inner_extrapolateMirror_dim<int64_t>(info.nd, a,b);
    case ca::t_uint8: 
      return inner_extrapolateMirror_dim<uint8_t>(info.nd, a,b);
    case ca::t_uint16:
      return inner_extrapolateMirror_dim<uint16_t>(info.nd, a,b);
    case ca::t_uint32: 
      return inner_extrapolateMirror_dim<uint32_t>(info.nd, a,b);
    case ca::t_uint64: 
      return inner_extrapolateMirror_dim<uint64_t>(info.nd, a,b);
    case ca::t_float32:
      return inner_extrapolateMirror_dim<float>(info.nd, a,b);
    case ca::t_float64: 
      return inner_extrapolateMirror_dim<double>(info.nd, a,b);
    case ca::t_complex64: 
      return inner_extrapolateMirror_dim<std::complex<float> >(info.nd, a,b);
    case ca::t_complex128: 
      return inner_extrapolateMirror_dim<std::complex<double> >(info.nd, a,b);
    default: PYTHON_ERROR(TypeError, "mirror extrapolation not supported for array with type '%s'", info.str().c_str());
  }
}

void bind_sp_extrapolate() {

  def("extrapolateConstant", &extrapolateConstant, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array with a constant, given a 1 or 2D input array.");

  def("extrapolateZero", &extrapolateZero, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array with zeros, given a 1 or 2D input array.");

  def("extrapolateNearest", &extrapolateNearest, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array with nearest values, given a 1 or 2D input array.");

  def("extrapolateCircular", &extrapolateCircular, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array using circular extrapolation, given a 1 or 2D input array.");

  def("extrapolateMirror", &extrapolateMirror, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array using mirroring, given a 1 or 2D input array.");

}
