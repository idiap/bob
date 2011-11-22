/**
 * @file python/ip/src/TanTriggs.cc
 * @date Fri Mar 18 18:09:25 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Tan and Triggs preprocessing filter to python
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

#include "ip/TanTriggs.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

static const char* ttdoc = "Objects of this class, after configuration, can preprocess images. It does this using the method described by Tan and Triggs in the paper titled \" Enhanced_Local_Texture_Feature_Sets for_Face_Recognition_Under_Difficult_Lighting_Conditions\", published in 2007";

template <typename T> static void inner_call1 (ip::TanTriggs& obj, 
    tp::const_ndarray input, tp::ndarray output) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  obj(input.bz<T,2>(), output_);
}

static void call1 (ip::TanTriggs& obj, tp::const_ndarray input,
    tp::ndarray output) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_call1<uint8_t>(obj, input, output);
    case ca::t_uint16:
      return inner_call1<uint16_t>(obj, input, output);
    case ca::t_float64: 
      return inner_call1<double>(obj, input, output);
    default: PYTHON_ERROR(TypeError, "Tan&Triggers filter does not support array with type '%s'", info.str().c_str());
  }
}

#define TANTRIGGS_CALL_DEF(T) \

void bind_ip_tantriggs() {
  class_<ip::TanTriggs, boost::shared_ptr<ip::TanTriggs> >("TanTriggs", ttdoc, init<optional<const double, const double, const double, const int, const double, const double> >((arg("gamma")="0.1", arg("sigma0")="1", arg("sigma1")="2", arg("size")="2", arg("threshold")="10.", arg("alpha")="0.1"), "Constructs a new Tan and Triggs filter."))
    //.add_property("test", &ip::TanTriggs::getTest, &ip::TanTriggs::setTest)
    .def("__call__", &call1, (arg("input"), arg("output")), "Call an object of this type to compute a preprocessed image.")
    ;
}
