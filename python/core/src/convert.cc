/**
 * @file python/core/src/convert.cc
 * @date Tue Mar 8 17:18:45 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Bindings for type conversion with re-calibration of values
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

#include <boost/version.hpp>

#include "core/python/ndarray.h"
#include "core/convert.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;
namespace tc = bob::core;

template <typename Tdst, typename Tsrc, int N>
static object inner_convert (tp::const_ndarray src,
    object dst_range, object src_range) {

  if (!TPY_ISNONE(src_range)) {
    Tsrc src_min = extract<Tsrc>(src_range[0]);
    Tsrc src_max = extract<Tsrc>(src_range[1]);
    if (!TPY_ISNONE(dst_range)) { //both src_range and dst_range are valid

      Tdst dst_min = extract<Tdst>(dst_range[0]);
      Tdst dst_max = extract<Tdst>(dst_range[1]);
      blitz::Array<Tdst,N> dst = tc::convert<Tdst,Tsrc>(src.bz<Tsrc,N>(),
          dst_min, dst_max, src_min, src_max);
      return object(dst); ///< must copy again
    }
    else { //only src_range is valid
      blitz::Array<Tdst,N> dst = 
        tc::convertFromRange<Tdst,Tsrc>(src.bz<Tsrc,N>(), src_min, src_max);
      return object(dst); ///< must copy again
    }
  }

  else {
    if (!TPY_ISNONE(dst_range)) { //only dst_range is valid
      Tdst dst_min = extract<Tdst>(dst_range[0]);
      Tdst dst_max = extract<Tdst>(dst_range[1]);
      blitz::Array<Tdst,N> dst = 
        tc::convertToRange<Tdst,Tsrc>(src.bz<Tsrc,N>(), dst_min, dst_max);
      return object(dst); ///< must copy again
    }
    else { //use all defaults
      blitz::Array<Tdst,N> dst = tc::convert<Tdst,Tsrc>(src.bz<Tsrc,N>());
      return object(dst); ///< must copy again
    }
  }

  PYTHON_ERROR(RuntimeError, "this is not supposed to happen with conversions -- debug me now!");
}

template <typename Tdst, typename Tsrc> 
static object convert_to_dim (tp::const_ndarray src,
    object dst_range, object src_range) {
  const ca::typeinfo& src_type = src.type();
  switch (src_type.nd) {
    case 1: return inner_convert<Tdst, Tsrc, 1>(src, dst_range, src_range);
    case 2: return inner_convert<Tdst, Tsrc, 2>(src, dst_range, src_range);
    case 3: return inner_convert<Tdst, Tsrc, 3>(src, dst_range, src_range);
    case 4: return inner_convert<Tdst, Tsrc, 4>(src, dst_range, src_range);
    default:
      PYTHON_ERROR(TypeError, "conversion does not support " SIZE_T_FMT " dimensions", src_type.nd);
  }
}

template <typename T> 
static object convert_to (tp::const_ndarray src, 
    object dst_range, object src_range) {
  const ca::typeinfo& src_type = src.type();
  switch (src_type.dtype) {
    case ca::t_bool:
      return convert_to_dim<T, bool>(src, dst_range, src_range);
    case ca::t_int8:
      return convert_to_dim<T, int8_t>(src, dst_range, src_range);
    case ca::t_int16:
      return convert_to_dim<T, int16_t>(src, dst_range, src_range);
    case ca::t_int32:
      return convert_to_dim<T, int32_t>(src, dst_range, src_range);
    case ca::t_int64:
      return convert_to_dim<T, int64_t>(src, dst_range, src_range);
    case ca::t_uint8:
      return convert_to_dim<T, uint8_t>(src, dst_range, src_range);
    case ca::t_uint16:
      return convert_to_dim<T, uint16_t>(src, dst_range, src_range);
    case ca::t_uint32:
      return convert_to_dim<T, uint32_t>(src, dst_range, src_range);
    case ca::t_uint64:
      return convert_to_dim<T, uint64_t>(src, dst_range, src_range);
    case ca::t_float32:
      return convert_to_dim<T, float>(src, dst_range, src_range);
    case ca::t_float64:
      return convert_to_dim<T, double>(src, dst_range, src_range);
    default:
      PYTHON_ERROR(TypeError, "conversion from '%s' is not supported", src_type.str().c_str());
  }
}

static void assert_2tuple_or_none (object o) {
  if (TPY_ISNONE(o)) return;

  //must be a tuple with 2 entries
  extract<tuple> t_ext(o);

  if (!t_ext.check()) {
    std::string s = extract<std::string>(str(o));
    PYTHON_ERROR(TypeError, "cannot extract a 2-tuple from '%s', if unsure just set this parameter to None", s.c_str());
  }

  tuple t = t_ext();
  if (len(t) != 2) {
    std::string s = extract<std::string>(str(t));
    PYTHON_ERROR(TypeError, "cannot extract a 2-tuple from '%s', if unsure just set this parameter to None", s.c_str());
  }
}

static object convert (tp::const_ndarray src, object dtype_like,
    object dst_range=object(), object src_range=object()) {

  //check input parameters
  assert_2tuple_or_none(dst_range);
  assert_2tuple_or_none(src_range);

  tp::dtype dst_type(dtype_like);
  switch (dst_type.eltype()) {
    case ca::t_uint8: 
      return convert_to<uint8_t>(src, dst_range, src_range);
    case ca::t_uint16:
      return convert_to<uint16_t>(src, dst_range, src_range);
    case ca::t_float64:
      return convert_to<double>(src, dst_range, src_range);
    default:
      PYTHON_ERROR(TypeError, "conversion to '%s' is not supported", dst_type.cxx_str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(convert_overloads, convert, 2, 4)

void bind_core_array_convert() {
  def("convert", &convert, convert_overloads((arg("array"), arg("dtype"), arg("destRange")=object(), arg("sourceRange")=object()), "Function which allows to convert/rescale a array of a given type into another array of a possibly different type with re-scaling. Typically, this can be used to rescale a 16 bit precision grayscale image (2D array) into an 8 bit precision grayscale image.\n\n  Parameters:\n\n    array -- (array) Input array\n\n    dtype -- (string) Controls the output element type for the returned array\n\n    destRrange -- (tuple) Determines the range to be deployed at the returned array\n\n    sourceRange -- (tuple) Determines the input range that will be used for the scaling\n\n  Returns: A new array with the same shape as this one, but re-scaled and with its element type as indicated by the user."));
}
