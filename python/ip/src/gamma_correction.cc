/**
 * @file python/ip/src/gamma_correction.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds gamma correction into python
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
#include "ip/gammaCorrection.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ip = bob::ip;
namespace ca = bob::core::array;

template <typename T, int N>
static void inner_gammaCorrection (tp::const_ndarray src, tp::ndarray dst,
    double g) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  ip::gammaCorrection<T>(src.bz<T,N>(), dst_, g);
}

static void gamma_correction (tp::const_ndarray src, tp::ndarray dst, double g) {
  const ca::typeinfo& info = src.type();

  if (info.nd != 2) PYTHON_ERROR(TypeError, "gamma correction does not support input of type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_gammaCorrection<uint8_t,2>(src, dst, g);
    case ca::t_uint16:
      return inner_gammaCorrection<uint16_t,2>(src, dst, g);
    case ca::t_float64:
      return inner_gammaCorrection<double,2>(src, dst, g);
    default:
      PYTHON_ERROR(TypeError, "gamma correction does not support type '%s'", info.str().c_str());
  }
}

void bind_ip_gamma_correction() {
  def("gamma_correction", &gamma_correction, (arg("src"), arg("dst"), arg("gamma")), "Perform a power-law gamma correction on a 2D blitz array/image.");
}
