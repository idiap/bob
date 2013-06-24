/**
 * @file ip/python/vldsift.cc
 * @date Mon Jan 23 20:46:07 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds VLFeat Dense SIFT features to python
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include "bob/python/ndarray.h"
#include "bob/ip/VLDSIFT.h"

using namespace boost::python;

static void call_vldsift_(bob::ip::VLDSIFT& op, bob::python::const_ndarray src, bob::python::ndarray dst) {
  const bob::core::array::typeinfo& info_s = src.type();  
  if(info_s.nd != 2) 
    PYTHON_ERROR(TypeError, "sift features extractor does not support input array with " SIZE_T_FMT "dimensions.", info_s.nd);
  if(info_s.dtype != bob::core::array::t_float32)
    PYTHON_ERROR(TypeError, "sift features does not support input array of type '%s'.", info_s.str().c_str());

  const bob::core::array::typeinfo& info_d = dst.type();  
  if(info_d.nd != 2)
    PYTHON_ERROR(TypeError, "sift features extractor does not support output array with " SIZE_T_FMT "dimensions.", info_s.nd);
  if(info_d.dtype != bob::core::array::t_float32)
    PYTHON_ERROR(TypeError, "sift features does not support output array of type '%s'.", info_d.str().c_str());

  blitz::Array<float,2> dst_ = dst.bz<float,2>();
  op(src.bz<float,2>(), dst_);
}

static object call_vldsift(bob::ip::VLDSIFT& op, bob::python::const_ndarray src) {
  const bob::core::array::typeinfo& info = src.type();  
  if(info.nd != 2) 
    PYTHON_ERROR(TypeError, "sift features extractor does not support input array with " SIZE_T_FMT "dimensions.", info.nd);
  if(info.dtype != bob::core::array::t_float32)
    PYTHON_ERROR(TypeError, "sift features does not support input array of type '%s'.", info.str().c_str());

  bob::python::ndarray dst(bob::core::array::t_float32, op.getNKeypoints(), op.getDescriptorSize());
  blitz::Array<float,2> dst_ = dst.bz<float,2>();
  op(src.bz<float,2>(), dst_);
  return dst.self();
}


void bind_ip_vldsift() 
{
  static const char* VLDSIFT_doc = "Computes dense SIFT features using the VLFeat library";

  class_<bob::ip::VLDSIFT, boost::shared_ptr<bob::ip::VLDSIFT> >("VLDSIFT", VLDSIFT_doc, init<const size_t, const size_t, optional<const size_t, const size_t> >((arg("height"), arg("width"), arg("step")=5, arg("block_size")=5), "Creates an object to compute dense SIFT features"))
    .def(init<bob::ip::VLDSIFT&>(args("other")))
    .def(self == self)
    .def(self != self)
    .add_property("height", &bob::ip::VLDSIFT::getHeight, &bob::ip::VLDSIFT::setHeight, "The height of the image to process")
    .add_property("width", &bob::ip::VLDSIFT::getWidth, &bob::ip::VLDSIFT::setWidth, "The width of the image to process")
    .add_property("step_y", &bob::ip::VLDSIFT::getStepY, &bob::ip::VLDSIFT::setStepY, "The step along the y-axis")
    .add_property("step_x", &bob::ip::VLDSIFT::getStepX, &bob::ip::VLDSIFT::setStepX, "The step along the x-axis")
    .add_property("block_size_y", &bob::ip::VLDSIFT::getBlockSizeY, &bob::ip::VLDSIFT::setBlockSizeY, "The block size along the y-axis")
    .add_property("block_size_x", &bob::ip::VLDSIFT::getBlockSizeX, &bob::ip::VLDSIFT::setBlockSizeX, "The block size along the y-axis")
    .add_property("use_flat_window", &bob::ip::VLDSIFT::getUseFlatWindow, &bob::ip::VLDSIFT::setUseFlatWindow, "Whether to use a flat window or not (to boost the processing time)")
    .add_property("window_size", &bob::ip::VLDSIFT::getWindowSize, &bob::ip::VLDSIFT::setWindowSize, "The window size")
    .def("forward", &call_vldsift_, (arg("self"), arg("src"), arg("dst")), "Computes the dense SIFT features from an input image, using the VLFeat library. Both input and output arrays should have the expected size.")
    .def("__call__", &call_vldsift_, (arg("self"), arg("src"), arg("dst")), "Computes the dense SIFT features from an input image, using the VLFeat library. Both input and output arrays should have the expected size.")
    .def("forward", &call_vldsift, (arg("self"), arg("src")), "Computes the dense SIFT features from an input image, using the VLFeat library. Returns the descriptors.")
    .def("__call__", &call_vldsift, (arg("self"), arg("src")), "Computes the dense SIFT features from an input image, using the VLFeat library. Returns the descriptors.")
    .def("get_n_keypoints", &bob::ip::VLDSIFT::getNKeypoints, "Returns the number of keypoints for the current parameters/image size.")
    .def("get_descriptor_size", &bob::ip::VLDSIFT::getDescriptorSize, "Returns the descriptor size for the current parameters.")
  ;
}
