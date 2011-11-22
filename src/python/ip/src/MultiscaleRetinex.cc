/**
 * @file python/ip/src/MultiscaleRetinex.cc
 * @date Mon May 2 10:01:08 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Multiscale Retinex algorith into python
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
#include "ip/MultiscaleRetinex.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

template <typename T> static void inner_call (ip::MultiscaleRetinex& obj, 
    tp::const_ndarray input, tp::ndarray output) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  obj(input.bz<T,2>(), output_);
}

static void py_call (ip::MultiscaleRetinex& obj, tp::const_ndarray input,
    tp::ndarray output) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: return inner_call<uint8_t>(obj, input, output);
    case ca::t_uint16: return inner_call<uint16_t>(obj, input, output);
    case ca::t_float64: return inner_call<double>(obj, input, output);
    default: PYTHON_ERROR(TypeError, "MultiscaleRetinex __call__ operator does not support array with type '%s'", info.str().c_str());
  }
}

void bind_ip_msr() {
	class_<ip::MultiscaleRetinex, boost::shared_ptr<ip::MultiscaleRetinex> >("MultiscaleRetinex", "Applies the Multiscale Retinex algorithm", init<optional<const size_t, const int, const int, const double> >((arg("n_scales")=1,arg("size_min")=1, arg("size_step")=1, arg("sigma")=5.), "Creates a MultiscaleRetinex object."))
		.def("__call__", &py_call, (arg("self"), arg("src"), arg("dst")), "Applies the Multiscale Retinex algorithm to an image of type uint8, uint16 or double")
		;
}
