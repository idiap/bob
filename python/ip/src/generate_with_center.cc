/**
 * @file python/ip/src/generate_with_center.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the generateWithCenter operation into python
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
#include "ip/generateWithCenter.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ip = bob::ip;
namespace ca = bob::core::array;

template <typename T, int N>
static void inner_gwc (tp::const_ndarray src, tp::ndarray dst, int y, int x) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  ip::generateWithCenter<T>(src.bz<T,N>(), dst_, y, x);
}

static void gwc (tp::const_ndarray src, tp::ndarray dst, int y, int x) {
  const ca::typeinfo& info = src.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "generate with center does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_gwc<uint8_t,2>(src, dst, y, x);
    case ca::t_uint16:
      return inner_gwc<uint16_t,2>(src, dst, y, x);
    case ca::t_float64:
      return inner_gwc<double,2>(src, dst, y, x);
    default:
      PYTHON_ERROR(TypeError, "generate with center does not support type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_gwc2 (tp::const_ndarray src, tp::const_ndarray smask, 
    tp::ndarray dst, tp::ndarray dmask, int y, int x) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  ip::generateWithCenter<T>(src.bz<T,N>(), smask.bz<bool,N>(), dst_, dmask_, y, x);
}

static void gwc2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask, int y, int x) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "generate with center does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_gwc2<uint8_t,2>(src, smask, dst, dmask, y, x);
    case ca::t_uint16:
      return inner_gwc2<uint16_t,2>(src, smask, dst, dmask, y, x);
    case ca::t_float64:
      return inner_gwc2<double,2>(src, smask, dst, dmask, y, x);
    default:
      PYTHON_ERROR(TypeError, "generate with center does not support type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static object inner_gwc_shape (tp::const_ndarray src, int y, int x) {
  return object(ip::getGenerateWithCenterShape<T>(src.bz<T,N>(), y, x));
}

static object gwc_shape (tp::const_ndarray src, int y, int x) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "generate with center does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_gwc_shape<uint8_t,2>(src, y, x);
    case ca::t_uint16:
      return inner_gwc_shape<uint16_t,2>(src, y, x);
    case ca::t_float64:
      return inner_gwc_shape<double,2>(src, y, x);
    default:
      PYTHON_ERROR(TypeError, "generate with center does not support type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static object inner_gwc_offset (tp::const_ndarray src, int y, int x) {
  return object(ip::getGenerateWithCenterOffset<T>(src.bz<T,N>(), y, x));
}

static object gwc_offset (tp::const_ndarray src, int y, int x) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "generate with center does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_gwc_offset<uint8_t,2>(src, y, x);
    case ca::t_uint16:
      return inner_gwc_offset<uint16_t,2>(src, y, x);
    case ca::t_float64:
      return inner_gwc_offset<double,2>(src, y, x);
    default:
      PYTHON_ERROR(TypeError, "generate with center does not support type '%s'", info.str().c_str());
  }
}

void bind_ip_generate_with_center() {

  def("generate_with_center", &gwc, (arg("src"), arg("dst"), arg("center_y"), arg("center_x")), "Extend a 2D blitz array/image, putting a given point in the center.");

  def("generate_with_center", &gwc2, (arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("center_y"), arg("center_x")), "Extend a 2D blitz array/image, putting a given point in the center, taking mask into account.");

  def("get_generate_with_center_shape", &gwc_shape, (arg("src"), arg("center_y"), arg("center_x")), "Return the shape of the output 2D blitz array/image, when calling generate_with_center which puts a given point of an image in the center.");
  
  def("get_generate_with_center_offset", &gwc_offset, (arg("src"), arg("center_y"), arg("center_x")), "Return the offset of the output 2D blitz array/image, when calling generate_with_center which puts a given point of an image in the center.");

}
