/**
 * @file python/ip/src/crop_shift.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @Sun 20 Nov 17:13:34 2011 CET
 * @brief Binds crop and shift operations into python
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
#include "ip/crop.h"
#include "ip/shift.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ip = Torch::ip;
namespace ca = Torch::core::array;

template <typename T, int N>
static void inner_crop1 (tp::const_ndarray src, tp::ndarray dst,
    int y, int x, int h, int w, bool allow_out, bool zero_out) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  ip::crop<T>(src.bz<T,N>(), dst_, y, x, h, w, allow_out, zero_out);
}

template <typename T>
static void inner_crop1_dim (tp::const_ndarray src, tp::ndarray dst,
    int y, int x, int h, int w, bool allow_out, bool zero_out) {
  const ca::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_crop1<T,2>(src, dst, y, x, h, w, allow_out, zero_out);
    case 3: return inner_crop1<T,3>(src, dst, y, x, h, w, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "cropping does not support type '%s'", info.str().c_str());
  }
}

static void crop1 (tp::const_ndarray src, tp::ndarray dst,
    int y, int x, int h, int w, bool allow_out=false, bool zero_out=false) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_crop1_dim<uint8_t>(src, dst, y,x,h,w, allow_out, zero_out);
    case ca::t_uint16:
      return inner_crop1_dim<uint16_t>(src, dst, y,x,h,w, allow_out, zero_out);
    case ca::t_float64:
      return inner_crop1_dim<double>(src, dst, y,x,h,w, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "cropping does not support type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(crop1_overloads, crop1, 6, 8)

template <typename T, int N>
static void inner_crop2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask,
    int y, int x, int h, int w, bool allow_out, bool zero_out) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  ip::crop<T>(src.bz<T,N>(), smask.bz<bool,N>(), dst_, dmask_, y, x, h, w, allow_out, zero_out);
}

template <typename T>
static void inner_crop2_dim (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask,
    int y, int x, int h, int w, bool allow_out, bool zero_out) {
  const ca::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_crop2<T,2>(src, smask, dst, dmask, y, x, h, w, allow_out, zero_out);
    case 3: return inner_crop2<T,3>(src, smask, dst, dmask, y, x, h, w, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "cropping does not support type '%s'", info.str().c_str());
  }
}

static void crop2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask,
    int y, int x, int h, int w, bool allow_out=false, bool zero_out=false) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_crop2_dim<uint8_t>(src, smask, dst, dmask, y,x,h,w, allow_out, zero_out);
    case ca::t_uint16:
      return inner_crop2_dim<uint16_t>(src, smask, dst, dmask, y,x,h,w, allow_out, zero_out);
    case ca::t_float64:
      return inner_crop2_dim<double>(src, smask, dst, dmask, y,x,h,w, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "cropping does not support type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(crop2_overloads, crop2, 8, 10)

template <typename T, int N>
static void inner_shift1 (tp::const_ndarray src, tp::ndarray dst,
    int y, int x, bool allow_out, bool zero_out) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  ip::shift<T>(src.bz<T,N>(), dst_, y, x, allow_out, zero_out);
}

template <typename T>
static void inner_shift1_dim (tp::const_ndarray src, tp::ndarray dst,
    int y, int x, bool allow_out, bool zero_out) {
  const ca::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_shift1<T,2>(src, dst, y, x, allow_out, zero_out);
    case 3: return inner_shift1<T,3>(src, dst, y, x, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "shifting does not support type '%s'", info.str().c_str());
  }
}

static void shift1 (tp::const_ndarray src, tp::ndarray dst,
    int y, int x, bool allow_out=false, bool zero_out=false) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_shift1_dim<uint8_t>(src, dst, y,x, allow_out, zero_out);
    case ca::t_uint16:
      return inner_shift1_dim<uint16_t>(src, dst, y,x, allow_out, zero_out);
    case ca::t_float64:
      return inner_shift1_dim<double>(src, dst, y,x, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "shifting does not support type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shift1_overloads, shift1, 4, 6)

template <typename T, int N>
static void inner_shift2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask,
    int y, int x, bool allow_out, bool zero_out) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  ip::shift<T>(src.bz<T,N>(), smask.bz<bool,N>(), dst_, dmask_, y, x, allow_out, zero_out);
}

template <typename T>
static void inner_shift2_dim (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask,
    int y, int x, bool allow_out, bool zero_out) {
  const ca::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_shift2<T,2>(src, smask, dst, dmask, y, x, allow_out, zero_out);
    case 3: return inner_shift2<T,3>(src, smask, dst, dmask, y, x, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "shifting does not support type '%s'", info.str().c_str());
  }
}

static void shift2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask,
    int y, int x, bool allow_out=false, bool zero_out=false) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_shift2_dim<uint8_t>(src, smask, dst, dmask, y,x, allow_out, zero_out);
    case ca::t_uint16:
      return inner_shift2_dim<uint16_t>(src, smask, dst, dmask, y,x, allow_out, zero_out);
    case ca::t_float64:
      return inner_shift2_dim<double>(src, smask, dst, dmask, y,x, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "shifting does not support type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shift2_overloads, shift2, 6, 8)

void bind_ip_crop_shift() {
  def("crop", &crop1, crop1_overloads((arg("src"), arg("dst"), arg("crop_y"), arg("crop_x"), arg("crop_h"), arg("crop_w"), arg("allow_out")=false, arg("zero_out")=false), "Crop a 2 or 3D array/image."));
  def("crop", &crop2, crop2_overloads((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("crop_y"), arg("crop_x"), arg("crop_h"), arg("crop_w"), arg("allow_out")=false, arg("zero_out")=false), "Crop a 2 or 3D array/image, taking mask into account."));
  def("shift", &shift1, shift1_overloads((arg("src"), arg("dst"), arg("shift_y"), arg("shift_x"), arg("allow_out")=false, arg("zero_out")=false), "Shift a 2 or 3D array/image."));
  def("shift", &shift2, shift2_overloads((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("shift_y"), arg("shift_x"), arg("allow_out")=false, arg("zero_out")=false), "Shift a 2 or 3D array/image, taking mask into account."));
}
