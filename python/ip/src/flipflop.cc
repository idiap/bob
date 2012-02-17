/**
 * @file python/ip/src/flipflop.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the flip and flop operations into python
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
#include "ip/flipflop.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ip = bob::ip;
namespace ca = bob::core::array;

template <typename T, int N>
static void inner_flip (tp::const_ndarray src, tp::ndarray dst) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  ip::flip<T>(src.bz<T,N>(), dst_);
}

template <typename T>
static void inner_flip_dim (tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_flip<T,2>(src, dst);
    case 3: return inner_flip<T,3>(src, dst);
    default:
      PYTHON_ERROR(TypeError, "image flipping does not support type '%s'", info.str().c_str());
  }
}

static void flip (tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_flip_dim<uint8_t>(src, dst);
    case ca::t_uint16:
      return inner_flip_dim<uint16_t>(src, dst);
    case ca::t_float64:
      return inner_flip_dim<double>(src, dst);
    default:
      PYTHON_ERROR(TypeError, "image flipping does not support type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_flop (tp::const_ndarray src, tp::ndarray dst) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  ip::flop<T>(src.bz<T,N>(), dst_);
}

template <typename T>
static void inner_flop_dim (tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_flop<T,2>(src, dst);
    case 3: return inner_flop<T,3>(src, dst);
    default:
      PYTHON_ERROR(TypeError, "image flopping does not support type '%s'", info.str().c_str());
  }
}

static void flop (tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_flop_dim<uint8_t>(src, dst);
    case ca::t_uint16:
      return inner_flop_dim<uint16_t>(src, dst);
    case ca::t_float64:
      return inner_flop_dim<double>(src, dst);
    default:
      PYTHON_ERROR(TypeError, "image flopping does not support type '%s'", info.str().c_str());
  }
}

void bind_ip_flipflop() {
  static const char* FLIP_DOC = "Flip a 2 or 3D array/image upside-down.";
  static const char* FLOP_DOC = "Flop a 2 or 3D array/image left-right.";
  def("flip", &flip, (arg("src"), arg("dst")), FLIP_DOC); 
  def("flop", &flop, (arg("src"), arg("dst")), FLOP_DOC); 
}
