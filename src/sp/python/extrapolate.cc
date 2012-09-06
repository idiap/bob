/**
 * @file sp/python/extrapolate.cc
 * @date Tue Sep 27 23:26:46 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds extrapolation to python
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "bob/core/python/ndarray.h"
#include "bob/sp/extrapolate.h"

using namespace boost::python;

template <typename T, int N>
static void inner_extrapolateConstant_dim_size(bob::python::const_ndarray a,
  bob::python::ndarray b, object c) 
{
  blitz::Array<T,N> b_ = b.bz<T,N>();
  bob::sp::extrapolateConstant<T>(a.bz<T,N>(), b_, extract<T>(c));
}

template <typename T>
static void inner_extrapolateConstant_dim(size_t nd, 
  bob::python::const_ndarray a, bob::python::ndarray b, object c) 
{
  switch (nd) {
    case 1: return inner_extrapolateConstant_dim_size<T,1>(a,b,c);
    case 2: return inner_extrapolateConstant_dim_size<T,2>(a,b,c);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate_constant not supported for array with " SIZE_T_FMT " dimensions.", nd);
  }
}

static void extrapolateConstant(bob::python::const_ndarray a, 
  bob::python::ndarray b, object c)
{
  const bob::core::array::typeinfo& info = a.type();
  switch (info.dtype) {
    case bob::core::array::t_bool: 
      return inner_extrapolateConstant_dim<bool>(info.nd, a,b,c);
    case bob::core::array::t_int8: 
      return inner_extrapolateConstant_dim<int8_t>(info.nd, a,b,c);
    case bob::core::array::t_int16: 
      return inner_extrapolateConstant_dim<int16_t>(info.nd, a,b,c);
    case bob::core::array::t_int32: 
      return inner_extrapolateConstant_dim<int32_t>(info.nd, a,b,c);
    case bob::core::array::t_int64: 
      return inner_extrapolateConstant_dim<int64_t>(info.nd, a,b,c);
    case bob::core::array::t_uint8: 
      return inner_extrapolateConstant_dim<uint8_t>(info.nd, a,b,c);
    case bob::core::array::t_uint16:
      return inner_extrapolateConstant_dim<uint16_t>(info.nd, a,b,c);
    case bob::core::array::t_uint32: 
      return inner_extrapolateConstant_dim<uint32_t>(info.nd, a,b,c);
    case bob::core::array::t_uint64: 
      return inner_extrapolateConstant_dim<uint64_t>(info.nd, a,b,c);
    case bob::core::array::t_float32:
      return inner_extrapolateConstant_dim<float>(info.nd, a,b,c);
    case bob::core::array::t_float64: 
      return inner_extrapolateConstant_dim<double>(info.nd, a,b,c);
    case bob::core::array::t_complex64: 
      return inner_extrapolateConstant_dim<std::complex<float> >(info.nd, a,b,c);
    case bob::core::array::t_complex128: 
      return inner_extrapolateConstant_dim<std::complex<double> >(info.nd, a,b,c);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate_constant not supported for array with type '%s'.", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_extrapolateZero_dim_size(bob::python::const_ndarray a,
  bob::python::ndarray b) 
{
  blitz::Array<T,N> b_ = b.bz<T,N>();
  bob::sp::extrapolateZero<T>(a.bz<T,N>(), b_);
}

template <typename T>
static void inner_extrapolateZero_dim(size_t nd, bob::python::const_ndarray a,
  bob::python::ndarray b) 
{
  switch (nd) {
    case 1: return inner_extrapolateZero_dim_size<T,1>(a,b);
    case 2: return inner_extrapolateZero_dim_size<T,2>(a,b);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate_zero not supported for array with " SIZE_T_FMT " dimensions.", nd);
  }
}

static void extrapolateZero(bob::python::const_ndarray a, bob::python::ndarray b)
{
  const bob::core::array::typeinfo& info = a.type();
  switch (info.dtype) {
    case bob::core::array::t_bool: 
      return inner_extrapolateZero_dim<bool>(info.nd, a,b);
    case bob::core::array::t_int8: 
      return inner_extrapolateZero_dim<int8_t>(info.nd, a,b);
    case bob::core::array::t_int16: 
      return inner_extrapolateZero_dim<int16_t>(info.nd, a,b);
    case bob::core::array::t_int32: 
      return inner_extrapolateZero_dim<int32_t>(info.nd, a,b);
    case bob::core::array::t_int64: 
      return inner_extrapolateZero_dim<int64_t>(info.nd, a,b);
    case bob::core::array::t_uint8: 
      return inner_extrapolateZero_dim<uint8_t>(info.nd, a,b);
    case bob::core::array::t_uint16:
      return inner_extrapolateZero_dim<uint16_t>(info.nd, a,b);
    case bob::core::array::t_uint32: 
      return inner_extrapolateZero_dim<uint32_t>(info.nd, a,b);
    case bob::core::array::t_uint64: 
      return inner_extrapolateZero_dim<uint64_t>(info.nd, a,b);
    case bob::core::array::t_float32:
      return inner_extrapolateZero_dim<float>(info.nd, a,b);
    case bob::core::array::t_float64: 
      return inner_extrapolateZero_dim<double>(info.nd, a,b);
    case bob::core::array::t_complex64: 
      return inner_extrapolateZero_dim<std::complex<float> >(info.nd, a,b);
    case bob::core::array::t_complex128: 
      return inner_extrapolateZero_dim<std::complex<double> >(info.nd, a,b);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate.zero not supported for array with type '%s'.", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_extrapolateNearest_dim_size(bob::python::const_ndarray a,
  bob::python::ndarray b) 
{
  blitz::Array<T,N> b_ = b.bz<T,N>();
  bob::sp::extrapolateNearest<T>(a.bz<T,N>(), b_);
}

template <typename T>
static void inner_extrapolateNearest_dim(size_t nd, bob::python::const_ndarray a,
  bob::python::ndarray b) 
{
  switch (nd) {
    case 1: return inner_extrapolateNearest_dim_size<T,1>(a,b);
    case 2: return inner_extrapolateNearest_dim_size<T,2>(a,b);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate_nearest not supported for array with " SIZE_T_FMT " dimensions.", nd);
  }
}

static void extrapolateNearest(bob::python::const_ndarray a, bob::python::ndarray b) 
{
  const bob::core::array::typeinfo& info = a.type();
  switch (info.dtype) {
    case bob::core::array::t_bool: 
      return inner_extrapolateNearest_dim<bool>(info.nd, a,b);
    case bob::core::array::t_int8: 
      return inner_extrapolateNearest_dim<int8_t>(info.nd, a,b);
    case bob::core::array::t_int16: 
      return inner_extrapolateNearest_dim<int16_t>(info.nd, a,b);
    case bob::core::array::t_int32: 
      return inner_extrapolateNearest_dim<int32_t>(info.nd, a,b);
    case bob::core::array::t_int64: 
      return inner_extrapolateNearest_dim<int64_t>(info.nd, a,b);
    case bob::core::array::t_uint8: 
      return inner_extrapolateNearest_dim<uint8_t>(info.nd, a,b);
    case bob::core::array::t_uint16:
      return inner_extrapolateNearest_dim<uint16_t>(info.nd, a,b);
    case bob::core::array::t_uint32: 
      return inner_extrapolateNearest_dim<uint32_t>(info.nd, a,b);
    case bob::core::array::t_uint64: 
      return inner_extrapolateNearest_dim<uint64_t>(info.nd, a,b);
    case bob::core::array::t_float32:
      return inner_extrapolateNearest_dim<float>(info.nd, a,b);
    case bob::core::array::t_float64: 
      return inner_extrapolateNearest_dim<double>(info.nd, a,b);
    case bob::core::array::t_complex64: 
      return inner_extrapolateNearest_dim<std::complex<float> >(info.nd, a,b);
    case bob::core::array::t_complex128: 
      return inner_extrapolateNearest_dim<std::complex<double> >(info.nd, a,b);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate_nearest not supported for array with type '%s'.", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_extrapolateCircular_dim_size(bob::python::const_ndarray a,
  bob::python::ndarray b) 
{
  blitz::Array<T,N> b_ = b.bz<T,N>();
  bob::sp::extrapolateCircular<T>(a.bz<T,N>(), b_);
}

template <typename T>
static void inner_extrapolateCircular_dim(size_t nd, bob::python::const_ndarray a,
  bob::python::ndarray b) 
{
  switch (nd) {
    case 1: return inner_extrapolateCircular_dim_size<T,1>(a,b);
    case 2: return inner_extrapolateCircular_dim_size<T,2>(a,b);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolation_circular not supported for array with " SIZE_T_FMT " dimensions.", nd);
  }
}

static void extrapolateCircular(bob::python::const_ndarray a, bob::python::ndarray b) 
{
  const bob::core::array::typeinfo& info = a.type();
  switch (info.dtype) {
    case bob::core::array::t_bool: 
      return inner_extrapolateCircular_dim<bool>(info.nd, a,b);
    case bob::core::array::t_int8: 
      return inner_extrapolateCircular_dim<int8_t>(info.nd, a,b);
    case bob::core::array::t_int16: 
      return inner_extrapolateCircular_dim<int16_t>(info.nd, a,b);
    case bob::core::array::t_int32: 
      return inner_extrapolateCircular_dim<int32_t>(info.nd, a,b);
    case bob::core::array::t_int64: 
      return inner_extrapolateCircular_dim<int64_t>(info.nd, a,b);
    case bob::core::array::t_uint8: 
      return inner_extrapolateCircular_dim<uint8_t>(info.nd, a,b);
    case bob::core::array::t_uint16:
      return inner_extrapolateCircular_dim<uint16_t>(info.nd, a,b);
    case bob::core::array::t_uint32: 
      return inner_extrapolateCircular_dim<uint32_t>(info.nd, a,b);
    case bob::core::array::t_uint64: 
      return inner_extrapolateCircular_dim<uint64_t>(info.nd, a,b);
    case bob::core::array::t_float32:
      return inner_extrapolateCircular_dim<float>(info.nd, a,b);
    case bob::core::array::t_float64: 
      return inner_extrapolateCircular_dim<double>(info.nd, a,b);
    case bob::core::array::t_complex64: 
      return inner_extrapolateCircular_dim<std::complex<float> >(info.nd, a,b);
    case bob::core::array::t_complex128: 
      return inner_extrapolateCircular_dim<std::complex<double> >(info.nd, a,b);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate_circular not supported for array with type '%s'.", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_extrapolateMirror_dim_size(bob::python::const_ndarray a,
    bob::python::ndarray b) {
  blitz::Array<T,N> b_ = b.bz<T,N>();
  bob::sp::extrapolateMirror<T>(a.bz<T,N>(), b_);
}

template <typename T>
static void inner_extrapolateMirror_dim(size_t nd, bob::python::const_ndarray a,
  bob::python::ndarray b) 
{
  switch (nd) {
    case 1: return inner_extrapolateMirror_dim_size<T,1>(a,b);
    case 2: return inner_extrapolateMirror_dim_size<T,2>(a,b);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate_mirror not supported for array with " SIZE_T_FMT " dimensions.", nd);
  }
}

static void extrapolateMirror(bob::python::const_ndarray a, bob::python::ndarray b) {
  const bob::core::array::typeinfo& info = a.type();
  switch (info.dtype) {
    case bob::core::array::t_bool: 
      return inner_extrapolateMirror_dim<bool>(info.nd, a,b);
    case bob::core::array::t_int8: 
      return inner_extrapolateMirror_dim<int8_t>(info.nd, a,b);
    case bob::core::array::t_int16: 
      return inner_extrapolateMirror_dim<int16_t>(info.nd, a,b);
    case bob::core::array::t_int32: 
      return inner_extrapolateMirror_dim<int32_t>(info.nd, a,b);
    case bob::core::array::t_int64: 
      return inner_extrapolateMirror_dim<int64_t>(info.nd, a,b);
    case bob::core::array::t_uint8: 
      return inner_extrapolateMirror_dim<uint8_t>(info.nd, a,b);
    case bob::core::array::t_uint16:
      return inner_extrapolateMirror_dim<uint16_t>(info.nd, a,b);
    case bob::core::array::t_uint32: 
      return inner_extrapolateMirror_dim<uint32_t>(info.nd, a,b);
    case bob::core::array::t_uint64: 
      return inner_extrapolateMirror_dim<uint64_t>(info.nd, a,b);
    case bob::core::array::t_float32:
      return inner_extrapolateMirror_dim<float>(info.nd, a,b);
    case bob::core::array::t_float64: 
      return inner_extrapolateMirror_dim<double>(info.nd, a,b);
    case bob::core::array::t_complex64: 
      return inner_extrapolateMirror_dim<std::complex<float> >(info.nd, a,b);
    case bob::core::array::t_complex128: 
      return inner_extrapolateMirror_dim<std::complex<double> >(info.nd, a,b);
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate_mirror not supported for array with type '%s'.", info.str().c_str());
  }
}


template <typename T, int N>
static void inner_extrapolate_dim_size(bob::python::const_ndarray a,
  bob::python::ndarray b, const bob::sp::Extrapolation::BorderType border_type, 
  const T val) 
{
  blitz::Array<T,N> b_ = b.bz<T,N>();
  bob::sp::extrapolate<T>(a.bz<T,N>(), b_, border_type, val);
}

template <typename T>
static void inner_extrapolate_dim(size_t nd, bob::python::const_ndarray a,
  bob::python::ndarray b, const bob::sp::Extrapolation::BorderType border_type, 
  const object val) 
{
  switch (nd) {
    case 1: inner_extrapolate_dim_size<T,1>(a,b,border_type,boost::python::extract<T>(val)); break;
    case 2: inner_extrapolate_dim_size<T,2>(a,b,border_type,boost::python::extract<T>(val)); break;
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate not supported for array with " SIZE_T_FMT " dimensions.", nd);
  }
}

static void extrapolate(bob::python::const_ndarray a, bob::python::ndarray b,
  const bob::sp::Extrapolation::BorderType border_type, const object val) 
{
  const bob::core::array::typeinfo& info = a.type();
  switch (info.dtype) {
    case bob::core::array::t_bool: 
      inner_extrapolate_dim<bool>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_int8: 
       inner_extrapolate_dim<int8_t>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_int16: 
       inner_extrapolate_dim<int16_t>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_int32: 
       inner_extrapolate_dim<int32_t>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_int64: 
       inner_extrapolate_dim<int64_t>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_uint8: 
       inner_extrapolate_dim<uint8_t>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_uint16:
       inner_extrapolate_dim<uint16_t>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_uint32: 
       inner_extrapolate_dim<uint32_t>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_uint64: 
       inner_extrapolate_dim<uint64_t>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_float32:
       inner_extrapolate_dim<float>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_float64: 
       inner_extrapolate_dim<double>(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_complex64: 
       inner_extrapolate_dim<std::complex<float> >(info.nd, a, b, border_type, val); break;
    case bob::core::array::t_complex128: 
       inner_extrapolate_dim<std::complex<double> >(info.nd, a, b, border_type, val); break;
    default: PYTHON_ERROR(TypeError, "bob.sp.extrapolate not supported for array with type '%s'.", info.str().c_str());
  }
}


void bind_sp_extrapolate() 
{
  enum_<bob::sp::Extrapolation::BorderType>("BorderType")
    .value("Zero", bob::sp::Extrapolation::Zero)
    .value("Constant", bob::sp::Extrapolation::Constant)
    .value("NearestNeighbour", bob::sp::Extrapolation::NearestNeighbour)
    .value("Circular", bob::sp::Extrapolation::Circular)
    .value("Mirror", bob::sp::Extrapolation::Mirror)
    ;
 
  def("extrapolate_constant", &extrapolateConstant, (arg("src"), arg("dst"), arg("constant")), "Extrapolates the values with a constant, given a 1D or 2D input array.");

  def("extrapolate_zero", &extrapolateZero, (arg("src"), arg("dst")), "Extrapolates the values with zeros, given a 1D or 2D input array.");

  def("extrapolate_nearest", &extrapolateNearest, (arg("src"), arg("dst")), "Extrapolates the values with nearest values, given a 1D or 2D input array.");

  def("extrapolate_circular", &extrapolateCircular, (arg("src"), arg("dst")), "Extrapolates the values using circular extrapolation, given a 1D or 2D input array.");

  def("extrapolate_mirror", &extrapolateMirror, (arg("src"), arg("dst")), "Extrapolates the values using mirroring, given a 1D or 2D input array.");

  def("extrapolate", &extrapolate, (arg("src"), arg("dst"), arg("border")=bob::sp::Extrapolation::Zero, arg("value")=0), "Extrapolates the values using the specified border type, given a 1D or 2D input array. The parameter value is only ever used if the border type is set to constant.");
}
