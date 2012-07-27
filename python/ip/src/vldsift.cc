/**
 * @file python/ip/src/vldsift.cc
 * @date Mon Jan 23 20:46:07 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds VLFeat Dense SIFT features to python
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
#include "ip/VLDSIFT.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ip = bob::ip;
namespace ca = bob::core::array;

static void call_vldsift_(ip::VLDSIFT& op, tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info_s = src.type();  
  if (info_s.nd != 2) PYTHON_ERROR(TypeError, "sift features extractor does not support input of type '%s'", info_s.str().c_str());
  if(info_s.dtype != ca::t_float32)
    PYTHON_ERROR(TypeError, "sift features does not support type '%s'", info_s.str().c_str());

  const ca::typeinfo& info_d = dst.type();  
  if (info_d.nd != 2) PYTHON_ERROR(TypeError, "sift features extractor does not support input of type '%s'", info_d.str().c_str());
  if(info_d.dtype != ca::t_float32)
    PYTHON_ERROR(TypeError, "sift features does not support type '%s'", info_d.str().c_str());

  blitz::Array<float,2> dst_ = dst.bz<float,2>();
  op(src.bz<float,2>(), dst_);
}

static object call_vldsift(ip::VLDSIFT& op, tp::const_ndarray src) {
  const ca::typeinfo& info = src.type();  
  if (info.nd != 2) PYTHON_ERROR(TypeError, "sift features extractor does not support input of type '%s'", info.str().c_str());
  if(info.dtype != ca::t_float32)
    PYTHON_ERROR(TypeError, "sift features does not support type '%s'", info.str().c_str());

  tp::ndarray dst(ca::t_float32, op.getNKeypoints(), op.getDescriptorSize());
  blitz::Array<float,2> dst_ = dst.bz<float,2>();
  op(src.bz<float,2>(), dst_);
  return dst.self();
}


void bind_ip_vldsift() {
  static const char* VLDSIFT_doc = "Computes dense SIFT features using the VLFeat library";

  class_<ip::VLDSIFT, boost::shared_ptr<ip::VLDSIFT> >("VLDSIFT", VLDSIFT_doc, init<const int, const int, optional<const int, const int> >((arg("height"), arg("width"), arg("step")=5, arg("bin_size")=5), "Creates an object to compute dense SIFT features"))
      .def("forward", &call_vldsift_, (arg("self"), arg("src"), arg("dst")), "Computes the dense SIFT features from an input image, using the VLFeat library. Both input and output arrays should have the expected size.")
      .def("__call__", &call_vldsift_, (arg("self"), arg("src"), arg("dst")), "Computes the dense SIFT features from an input image, using the VLFeat library. Both input and output arrays should have the expected size.")
      .def("forward", &call_vldsift, (arg("self"), arg("src")), "Computes the dense SIFT features from an input image, using the VLFeat library. Returns the descriptors.")
      .def("__call__", &call_vldsift, (arg("self"), arg("src")), "Computes the dense SIFT features from an input image, using the VLFeat library. Returns the descriptors.")
      .def("get_n_keypoints", &ip::VLDSIFT::getNKeypoints, "Returns the number of keypoints for the current parameters/image size.")
      .def("get_descriptor_size", &ip::VLDSIFT::getDescriptorSize, "Returns the descriptor size for the current parameters.")
    ;
}
