/**
 * @file ip/python/FaceEyesNorm.cc
 * @date Fri Apr 15 18:44:41 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the FaceEyesNorm class to python
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

#include <bob/python/ndarray.h>
#include <bob/ip/FaceEyesNorm.h>

using namespace boost::python;

static const char* faceeyesnorm_doc = "Objects of this class, after configuration, can extract and normalize faces, given their eye center coordinates.";

template <typename T> 
static void inner_call1(bob::ip::FaceEyesNorm& obj, 
  bob::python::const_ndarray input, bob::python::ndarray output,
  double e1y, double e1x, double e2y, double e2x)
{
  blitz::Array<double,2> output_ = output.bz<double,2>();
  obj(input.bz<T,2>(), output_, e1y, e1x, e2y, e2x);
}

static void call1(bob::ip::FaceEyesNorm& obj, bob::python::const_ndarray input,
    bob::python::ndarray output, double e1y, double e1x, double e2y, double e2x) 
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_call1<uint8_t>(obj, input, output, e1y, e1x, e2y, e2x);
    case bob::core::array::t_uint16:
      return inner_call1<uint16_t>(obj, input, output, e1y, e1x, e2y, e2x);
    case bob::core::array::t_float64: 
      return inner_call1<double>(obj, input, output, e1y, e1x, e2y, e2x);
    default: PYTHON_ERROR(TypeError, "FaceEyesNorm __call__ does not support array of type '%s'.", info.str().c_str());
  }
}

template <typename T> 
static object inner_call1b(bob::ip::FaceEyesNorm& op, 
  bob::python::const_ndarray src, double e1y, double e1x, double e2y, double e2x)
{
  bob::python::ndarray dst(bob::core::array::t_float64, op.getCropHeight(), 
    op.getCropWidth());
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<T,2>(), dst_, e1y, e1x, e2y, e2x);
  return dst.self();
}

static object call1b(bob::ip::FaceEyesNorm& op, bob::python::const_ndarray src,
  double e1y, double e1x, double e2y, double e2x)
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_call1b<uint8_t>(op, src, e1y, e1x, e2y, e2x);
    case bob::core::array::t_uint16:
      return inner_call1b<uint16_t>(op, src, e1y, e1x, e2y, e2x);
    case bob::core::array::t_float64: 
      return inner_call1b<double>(op, src, e1y, e1x, e2y, e2x);
    default: PYTHON_ERROR(TypeError, "FaceEyesNorm __call__ does not support array of type '%s'.", info.str().c_str());
  }
}

template <typename T> static void inner_call2(bob::ip::FaceEyesNorm& obj, 
  bob::python::const_ndarray input, bob::python::const_ndarray input_mask,
  bob::python::ndarray output, bob::python::ndarray output_mask,
  double e1y, double e1x, double e2y, double e2x)
{
  blitz::Array<double,2> output_ = output.bz<double,2>();
  blitz::Array<bool,2> output_mask_ = output_mask.bz<bool,2>();
  obj(input.bz<T,2>(), input_mask.bz<bool,2>(), output_, output_mask_,
      e1y, e1x, e2y, e2x);
}

static void call2(bob::ip::FaceEyesNorm& obj, bob::python::const_ndarray input,
  bob::python::const_ndarray input_mask, bob::python::ndarray output, 
  bob::python::ndarray output_mask, double e1y, double e1x, double e2y, double e2x) 
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_call2<uint8_t>(obj, input, input_mask, output, output_mask, e1y, e1x, e2y, e2x);
    case bob::core::array::t_uint16:
      return inner_call2<uint16_t>(obj, input, input_mask, output, output_mask, e1y, e1x, e2y, e2x);
    case bob::core::array::t_float64: 
      return inner_call2<double>(obj, input, input_mask, output, output_mask, e1y, e1x, e2y, e2x);
    default: PYTHON_ERROR(TypeError, "FaceEyesNorm __call__ does not support array of type '%s'.", info.str().c_str());
  }
}

void bind_ip_faceeyesnorm() {
  class_<bob::ip::FaceEyesNorm, boost::shared_ptr<bob::ip::FaceEyesNorm> >("FaceEyesNorm", faceeyesnorm_doc, init<const double, const size_t, const size_t, const double, const double>((arg("self"), arg("eyes_distance"), arg("crop_height"), arg("crop_width"), arg("crop_eyecenter_offset_h"), arg("crop_eyecenter_offset_w")), "Constructs a FaceEyeNorm object."))
      .def(init<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>(args("self", "crop_height", "crop_width", "re_y", "re_x", "le_y", "le_x"), "Creates a FaceEyesNorm class that will put the eyes to the given locations and crop the image to the desired size."))
      .def(init<bob::ip::FaceEyesNorm&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .add_property("eyes_distance", &bob::ip::FaceEyesNorm::getEyesDistance, &bob::ip::FaceEyesNorm::setEyesDistance, "Expected distance between the eyes after the geometric normalization.")
      .add_property("crop_height", &bob::ip::FaceEyesNorm::getCropHeight, &bob::ip::FaceEyesNorm::setCropHeight, "Height of the cropping area after the geometric normalization.")
      .add_property("crop_width", &bob::ip::FaceEyesNorm::getCropWidth, &bob::ip::FaceEyesNorm::setCropWidth, "Width of the cropping area after the geometric normalization.")
      .add_property("crop_offset_h", &bob::ip::FaceEyesNorm::getCropOffsetH, &bob::ip::FaceEyesNorm::setCropOffsetH, "y-coordinate of the point in the cropping area which is the middle of the segment defined by the eyes after the geometric normalization.")
      .add_property("crop_offset_w", &bob::ip::FaceEyesNorm::getCropOffsetW, &bob::ip::FaceEyesNorm::setCropOffsetW, "x-coordinate of the point in the cropping area which is the middle of the segment defined by the eyes after the geometric normalization.")
      .add_property("last_angle", &bob::ip::FaceEyesNorm::getLastAngle, "The angle value (in degrees) used by the rotation involved in the last call of the operator ()")
      .add_property("last_scale", &bob::ip::FaceEyesNorm::getLastScale, "The scaling factor used by the scaling involved in the last call of the operator ()")
      .def("__call__", &call1, (arg("self"), arg("input"), arg("output"), arg("re_y"), arg("re_x"), arg("le_y"), arg("le_x")), "Extracts a face given the coordinates of the left (le_y, le_x) and right (re_y, re_x) eye centers. Please note that the horizontal position le_x of the left eye is usually larger than the position re_x of the right eye.")
      .def("__call__", &call1b, (arg("self"), arg("input"), arg("re_y"), arg("re_x"), arg("le_y"), arg("le_x")), "Extracts a face given the coordinates of the left (le_y, le_x) and right (re_y, re_x) eye centers. Please note that the horizontal position le_x of the left eye is usually larger than the position re_x of the right eye. The output is allocated and returned.")
      .def("__call__", &call2, (arg("self"), arg("input"), arg("input_mask"), arg("output"), arg("output_mask"), arg("re_y"), arg("re_x"), arg("le_y"), arg("le_x")), "Extracts a face given the coordinates of the left (le_y, le_x) and right (re_y, re_x) eye centers, taking mask into account.")
    ;
}
