/**
 * @file python/ip/src/GeomNorm.cc
 * @date Mon Apr 11 23:07:02 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the GeomNorm class to python
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

#include "bob/core/python/ndarray.h"
#include "bob/ip/GeomNorm.h"
#include "bob/ip/maxRectInMask.h"

using namespace boost::python;

static const char* GEOMNORM_DOC = "Objects of this class, after configuration, can perform a geometric normalization.";
static const char* MAXRECTINMASK2D_DOC = "Given a 2D mask (a 2D blitz array of booleans), compute the maximum rectangle which only contains true values.";

template <typename T> 
static void inner_call1(bob::ip::GeomNorm& obj, 
  bob::python::const_ndarray input, bob::python::ndarray output,
  const double a, const double b)
{
  blitz::Array<double,2> output_ = output.bz<double,2>();
  obj(input.bz<T,2>(), output_, a,b);
}

static void call1(bob::ip::GeomNorm& obj, bob::python::const_ndarray input,
  bob::python::ndarray output, const double a, const double b)
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) 
  {
    case bob::core::array::t_uint8: 
      inner_call1<uint8_t>(obj, input, output, a,b);
      break;
    case bob::core::array::t_uint16:
      inner_call1<uint16_t>(obj, input, output, a,b);
      break;
    case bob::core::array::t_float64: 
      inner_call1<double>(obj, input, output, a,b);
      break;
    default: PYTHON_ERROR(TypeError, "geometric normalization does not support array with type '%s'", info.str().c_str());
  }
}

template <typename T> 
static void inner_call2(bob::ip::GeomNorm& obj, 
  bob::python::const_ndarray input, bob::python::const_ndarray input_mask,
  bob::python::ndarray output, bob::python::ndarray output_mask,
  const double a, const double b)
{
  blitz::Array<double,2> output_ = output.bz<double,2>();
  blitz::Array<bool,2> output_mask_ = output_mask.bz<bool,2>();
  obj(input.bz<T,2>(), input_mask.bz<bool,2>(), output_, output_mask_,
      a, b);
}

static void call2(bob::ip::GeomNorm& obj, bob::python::const_ndarray input,
  bob::python::const_ndarray input_mask, bob::python::ndarray output, bob::python::ndarray output_mask,
  const double a, const double b)
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) 
  {
    case bob::core::array::t_uint8: 
      inner_call2<uint8_t>(obj, input, input_mask, output, output_mask, a, b);
      break;
    case bob::core::array::t_uint16:
      inner_call2<uint16_t>(obj, input, input_mask, output, output_mask, a, b);
      break;
    case bob::core::array::t_float64: 
      inner_call2<double>(obj, input, input_mask, output, output_mask, a, b);
      break;
    default: PYTHON_ERROR(TypeError, "geometric normalization (with masks) does not support array with type '%s'", info.str().c_str());
  }
}

static blitz::TinyVector<double,2> call3(bob::ip::GeomNorm& obj,
  const blitz::TinyVector<double,2>& position, const double a, const double b)
{
  return obj(position,a,b);
}

void bind_ip_geomnorm() 
{
  class_<bob::ip::GeomNorm, boost::shared_ptr<bob::ip::GeomNorm> >("GeomNorm", GEOMNORM_DOC, init<const double, const double, const size_t, const size_t, const double, const double>((arg("rotation_angle"), arg("scaling_factor"), arg("crop_height"), arg("crop_width"), arg("crop_offset_h"), arg("crop_offset_w")), "Constructs a GeomNorm object."))
    .def(init<bob::ip::GeomNorm&>(args("other")))
    .def(self == self)
    .def(self != self)
    .add_property("rotation_angle", &bob::ip::GeomNorm::getRotationAngle, &bob::ip::GeomNorm::setRotationAngle, "Rotation angle for the geometric normalization (in radians)")
    .add_property("scaling_factor", &bob::ip::GeomNorm::getScalingFactor, &bob::ip::GeomNorm::setScalingFactor, "Scaling factor for the geometric normalization")
    .add_property("crop_height", &bob::ip::GeomNorm::getCropHeight, &bob::ip::GeomNorm::setCropHeight, "Height of the cropping area/output after the geometric normalization")
    .add_property("crop_width", &bob::ip::GeomNorm::getCropWidth, &bob::ip::GeomNorm::setCropWidth, "Width of the cropping area/output after the geometric normalization")
    .add_property("crop_offset_h", &bob::ip::GeomNorm::getCropOffsetH, &bob::ip::GeomNorm::setCropOffsetH, "y-coordinate of the rotation center in the new cropped area")
    .add_property("crop_offset_w", &bob::ip::GeomNorm::getCropOffsetW, &bob::ip::GeomNorm::setCropOffsetW, "x-coordinate of the rotation center in the new cropped area")
    .def("__call__", &call1, (arg("input"), arg("output"), arg("rotation_center_y"), arg("rotation_center_x")), "Call an object of this type to perform a geometric normalization of an image wrt. the given rotation center")
    .def("__call__", &call2, (arg("input"), arg("input_mask"), arg("output"), arg("output_mask"), arg("rotation_center_y"), arg("rotation_center_x")), "Call an object of this type to perform a geometric normalization of an image wrt. the given rotation center, taking mask into account.")
    .def("__call__", &call3, (arg("input"), arg("rotation_center_y"), arg("rotation_center_x")), "This function performs the geometric normalization for the given input position")
  ;

  def("max_rect_in_mask", (const blitz::TinyVector<int,4> (*)(const blitz::Array<bool,2>&))&bob::ip::maxRectInMask, (("src")), MAXRECTINMASK2D_DOC); 
}
