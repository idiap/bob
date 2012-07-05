/**
 * @file python/ip/src/FaceEyesNorm.cc
 * @date Fri Apr 15 18:44:41 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the FaceEyesNorm class to python
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

#include "ip/FaceEyesNorm.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace ip = bob::ip;
namespace tp = bob::python;
namespace ca = bob::core::array;

static const char* faceeyesnorm_doc = "Objects of this class, after configuration, can extract and normalize faces, given their eye center coordinates.";

template <typename T> static void inner_call1 (ip::FaceEyesNorm& obj, 
    tp::const_ndarray input, tp::ndarray output,
    int e1y, int e1x, int e2y, int e2x) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  obj(input.bz<T,2>(), output_, e1y, e1x, e2y, e2x);
}

static void call1 (ip::FaceEyesNorm& obj, tp::const_ndarray input,
    tp::ndarray output, int e1y, int e1x, int e2y, int e2x) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_call1<uint8_t>(obj, input, output, e1y, e1x, e2y, e2x);
    case ca::t_uint16:
      return inner_call1<uint16_t>(obj, input, output, e1y, e1x, e2y, e2x);
    case ca::t_float64: 
      return inner_call1<double>(obj, input, output, e1y, e1x, e2y, e2x);
    default: PYTHON_ERROR(TypeError, "face normalization does not support array with type '%s'", info.str().c_str());
  }
}

template <typename T> static void inner_call2 (ip::FaceEyesNorm& obj, 
    tp::const_ndarray input, tp::const_ndarray input_mask,
    tp::ndarray output, tp::ndarray output_mask,
    int e1y, int e1x, int e2y, int e2x) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  blitz::Array<bool,2> output_mask_ = output_mask.bz<bool,2>();
  obj(input.bz<T,2>(), input_mask.bz<bool,2>(), output_, output_mask_,
      e1y, e1x, e2y, e2x);
}

static void call2 (ip::FaceEyesNorm& obj, tp::const_ndarray input,
    tp::const_ndarray input_mask, tp::ndarray output, tp::ndarray output_mask,
    int e1y, int e1x, int e2y, int e2x) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_call2<uint8_t>(obj, input, input_mask, output, output_mask, e1y, e1x, e2y, e2x);
    case ca::t_uint16:
      return inner_call2<uint16_t>(obj, input, input_mask, output, output_mask, e1y, e1x, e2y, e2x);
    case ca::t_float64: 
      return inner_call2<double>(obj, input, input_mask, output, output_mask, e1y, e1x, e2y, e2x);
    default: PYTHON_ERROR(TypeError, "face normalization (with masks) does not support array with type '%s'", info.str().c_str());
  }
}

void bind_ip_faceeyesnorm() {
  class_<ip::FaceEyesNorm, boost::shared_ptr<ip::FaceEyesNorm> >("FaceEyesNorm", faceeyesnorm_doc, init<const int, const int, const int, const int, const int>((arg("eyes_distance"), arg("crop_height"), arg("crop_width"), arg("crop_eyecenter_offset_h"), arg("crop_eyecenter_offset_w")), "Constructs a FaceEyeNorm object."))
    .add_property("eyes_distance", &ip::FaceEyesNorm::getEyesDistance, &ip::FaceEyesNorm::setEyesDistance)
    .add_property("crop_height", &ip::FaceEyesNorm::getCropHeight, &ip::FaceEyesNorm::setCropHeight)
    .add_property("crop_width", &ip::FaceEyesNorm::getCropWidth, &ip::FaceEyesNorm::setCropWidth)
    .add_property("crop_offset_h", &ip::FaceEyesNorm::getCropOffsetH, &ip::FaceEyesNorm::setCropOffsetH)
    .add_property("crop_offset_w", &ip::FaceEyesNorm::getCropOffsetW, &ip::FaceEyesNorm::setCropOffsetW)
    .def("__call__", &call1, (arg("input"), arg("output"), arg("re_y"), arg("re_x"), arg("le_y"), arg("le_x")), "Call an object of this type to extract a face given the coordinates of the left (le_y, le_x) and right (re_y, re_x) eye centers. Please note that the horizontal position le_x of the left eye is usually larger than the position re_x of the right eye.")
    .def("__call__", &call2, (arg("input"), arg("input_mask"), arg("output"), arg("output_mask"), arg("re_y"), arg("re_x"), arg("le_y"), arg("le_x")), "Call an object of this type to extract a face given the coordinates of the left (le_y, le_x) and right (re_y, re_x) eye centers, taking mask into account.")
    ;
}
