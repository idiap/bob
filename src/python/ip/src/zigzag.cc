/**
 * @file python/ip/src/zigzag.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the zigzag operation into python
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
#include "ip/zigzag.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ip = Torch::ip;
namespace ca = Torch::core::array;

template <typename T>
static void inner_zigzag(tp::const_ndarray src, tp::ndarray dst, int nc,
    bool rf) {
  blitz::Array<T,1> dst_ = dst.bz<T,1>();
  ip::zigzag<T>(src.bz<T,2>(), dst_, nc, rf);
}

static void zigzag (tp::const_ndarray src, tp::ndarray dst,
    int nc=0, bool rf=false) {
  
  const ca::typeinfo& info = src.type();
  
  if (info.nd != 2) PYTHON_ERROR(TypeError, "zigzag does not support input of type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: return inner_zigzag<uint8_t>(src, dst, nc, rf);
    case ca::t_uint16: return inner_zigzag<uint16_t>(src, dst, nc, rf);
    case ca::t_float64: return inner_zigzag<double>(src, dst, nc, rf);
    default: PYTHON_ERROR(TypeError, "zigzag does not support type '%s'", info.str().c_str());
  }

}

BOOST_PYTHON_FUNCTION_OVERLOADS(zigzag_overloads, zigzag, 2, 4)

void bind_ip_zigzag() {
  def("zigzag", &zigzag, zigzag_overloads((arg("src"), arg("dst"), arg("n_coef")=0, arg("right_first")=false), "Extract a 1D array using a zigzag pattern from a 2D array/image."));
}
