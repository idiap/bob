/**
 * @file python/ip/src/gaussian.cc
 * @date Wed Apr 27 10:25:00 2011 +0200
 * @author Niklas Johansson <niklas.johansson@idiap.ch>
 *
 * @brief Binds Gaussian smoothing to python
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
#include "ip/Gaussian.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ip = bob::ip;
namespace ca = bob::core::array;

template <typename T, int N>
static void inner_call_gs (ip::Gaussian& op, tp::const_ndarray src, tp::ndarray dst) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  op(src.bz<T,N>(), dst_);
}

static void call_gs (ip::Gaussian& op, tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  
  if (info.nd != 2) PYTHON_ERROR(TypeError, "gaussian smoothing does not support input of type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: return inner_call_gs<uint8_t,2>(op, src, dst);
    case ca::t_uint16: return inner_call_gs<uint16_t,2>(op, src, dst);
    case ca::t_float64: return inner_call_gs<double,2>(op, src, dst);
    default:
      PYTHON_ERROR(TypeError, "gaussian smoothing does not support type '%s'", info.str().c_str());
  }
}

void bind_ip_gaussian() {
  static const char* gaussiandoc = "Performs gaussian smoothing";

	class_<ip::Gaussian, boost::shared_ptr<ip::Gaussian> >("Gaussian", gaussiandoc, init<optional<const int, const int, const double, const double> >((arg("radius_y")=1, arg("radius_x")=1, arg("sigma_y")=5., arg("sigma_x")=5.), "Creates a gaussian smoother"))
		.def("__call__", &call_gs, (arg("self"), arg("src"), arg("dst")), "Smooth an image")
		;
}
