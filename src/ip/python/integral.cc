/**
 * @file python/ip/src/integral.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds integral image implementation to python
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

#include "core/python/ndarray.h"
#include "ip/integral.h"

using namespace boost::python;

template <typename T, typename U, int N>
static void inner_integral (bob::python::const_ndarray src, bob::python::ndarray dst, bool b) {
  blitz::Array<U,N> dst_ = dst.bz<U,N>();
  bob::ip::integral(src.bz<T,N>(), dst_, b);
}

template <typename T, int N>
static void integral2 (bob::python::const_ndarray src, bob::python::ndarray dst, bool b) {
  const bob::core::array::typeinfo& info = dst.type();

  if(info.nd != 2)
    PYTHON_ERROR(TypeError, "integral image operator does not support output with " SIZE_T_FMT " dimensions.", info.nd);

  switch (info.dtype) {
    case bob::core::array::t_int8: return inner_integral<T,int8_t,N>(src, dst, b);
    case bob::core::array::t_int16: return inner_integral<T,int16_t,N>(src, dst, b);
    case bob::core::array::t_int32: return inner_integral<T,int32_t,N>(src, dst, b);
    case bob::core::array::t_int64: return inner_integral<T,int64_t,N>(src, dst, b);
    case bob::core::array::t_uint8: return inner_integral<T,uint8_t,N>(src, dst, b);
    case bob::core::array::t_uint16: return inner_integral<T,uint16_t,N>(src, dst, b);
    case bob::core::array::t_uint32: return inner_integral<T,uint32_t,N>(src, dst, b);
    case bob::core::array::t_uint64: return inner_integral<T,uint64_t,N>(src, dst, b);
    case bob::core::array::t_float32: return inner_integral<T,float,N>(src, dst, b);
    case bob::core::array::t_float64: return inner_integral<T,double,N>(src, dst, b);
    default:
      PYTHON_ERROR(TypeError, "integral image operator does not support output type '%s'", info.str().c_str());
  }

}

static void integral (bob::python::const_ndarray src, bob::python::ndarray dst, bool b=false) {
  const bob::core::array::typeinfo& info = src.type();

  if(info.nd != 2)
    PYTHON_ERROR(TypeError, "integral image operator does not support input with " SIZE_T_FMT " dimensions.", info.nd);

  switch (info.dtype) {
    case bob::core::array::t_uint8: return integral2<uint8_t,2>(src, dst, b);
    case bob::core::array::t_uint16: return integral2<uint16_t,2>(src, dst, b);
    case bob::core::array::t_float64: return integral2<double,2>(src, dst, b);
    default:
      PYTHON_ERROR(TypeError, "integral image operator does not support input type '%s'", info.str().c_str());
  }

}

BOOST_PYTHON_FUNCTION_OVERLOADS(integral_overloads, integral, 2, 3)

void bind_ip_integral() {
  def(BOOST_PP_STRINGIZE(integral), &integral, integral_overloads((arg("src"), arg("dst"), arg("add_zero_border")=false), "Compute the integral image of a 2D blitz array (image). It is the responsibility of the user to select an appropriate type for the numpy array which will contain the integral image. By default, src and dst should have the same size. If add_zero_border is set to true, then dst should be one pixel larger than src in each dimension.")); 
}
