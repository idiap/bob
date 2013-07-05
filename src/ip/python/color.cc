/**
 * @file ip/python/color.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds color converters to python
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

#include "bob/ip/color.h"
#include "bob/python/ndarray.h"

using namespace boost::python;

// RGB -> gray and vice versa
static object rgb_to_gray(const object& r, const object& g, const object& b, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t y;
        bob::ip::rgb_to_gray_one((uint8_t)extract<uint8_t>(r), (uint8_t)extract<uint8_t>(g), (uint8_t)extract<uint8_t>(b), y);
        return object(y);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t y;
        bob::ip::rgb_to_gray_one((uint16_t)extract<uint16_t>(r), (uint16_t)extract<uint16_t>(g), (uint16_t)extract<uint16_t>(b), y);
        return object(y);
      }
    case bob::core::array::t_float64:
      {
        double y;
        bob::ip::rgb_to_gray_one((double)extract<double>(r), (double)extract<double>(g), (double)extract<double>(b), y);
        return object(y);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}

static object gray_to_rgb(const object& y, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t r, g, b;
        bob::ip::gray_to_rgb_one((uint8_t)extract<uint8_t>(y), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t r, g, b;
        bob::ip::gray_to_rgb_one((uint16_t)extract<uint8_t>(y), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_float64:
      {
        double r, g, b;
        bob::ip::gray_to_rgb_one((double)extract<uint8_t>(y), r, g, b);
        return make_tuple(r, g, b);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}


// RGB to HSV and vice versa
static object rgb_to_hsv(const object& r, const object& g, const object& b, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t h, s, v;
        bob::ip::rgb_to_hsv_one((uint8_t)extract<uint8_t>(r), (uint8_t)extract<uint8_t>(g), (uint8_t)extract<uint8_t>(b), h, s, v);
        return make_tuple(h, s, v);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t h, s, v;
        bob::ip::rgb_to_hsv_one((uint16_t)extract<uint16_t>(r), (uint16_t)extract<uint16_t>(g), (uint16_t)extract<uint16_t>(b), h, s, v);
        return make_tuple(h, s, v);
      }
    case bob::core::array::t_float64:
      {
        double h, s, v;
        bob::ip::rgb_to_hsv_one((double)extract<double>(r), (double)extract<double>(g), (double)extract<double>(b), h, s, v);
        return make_tuple(h, s, v);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}

static object hsv_to_rgb(const object& h, const object& s, const object& v, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t r, g, b;
        bob::ip::hsv_to_rgb_one((uint8_t)extract<uint8_t>(h), (uint8_t)extract<uint8_t>(s), (uint8_t)extract<uint8_t>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t r, g, b;
        bob::ip::hsv_to_rgb_one((uint16_t)extract<uint16_t>(h), (uint16_t)extract<uint16_t>(s), (uint16_t)extract<uint16_t>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_float64:
      {
        double r, g, b;
        bob::ip::hsv_to_rgb_one((double)extract<double>(h), (double)extract<double>(s), (double)extract<double>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}


// RGB to HSL and vice versa
static object rgb_to_hsl(const object& r, const object& g, const object& b, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t h, s, l;
        bob::ip::rgb_to_hsl_one((uint8_t)extract<uint8_t>(r), (uint8_t)extract<uint8_t>(g), (uint8_t)extract<uint8_t>(b), h, s, l);
        return make_tuple(h, s, l);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t h, s, l;
        bob::ip::rgb_to_hsl_one((uint16_t)extract<uint16_t>(r), (uint16_t)extract<uint16_t>(g), (uint16_t)extract<uint16_t>(b), h, s, l);
        return make_tuple(h, s, l);
      }
    case bob::core::array::t_float64:
      {
        double h, s, l;
        bob::ip::rgb_to_hsl_one((double)extract<double>(r), (double)extract<double>(g), (double)extract<double>(b), h, s, l);
        return make_tuple(h, s, l);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}

static object hsl_to_rgb(const object& h, const object& s, const object& l, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t r, g, b;
        bob::ip::hsl_to_rgb_one((uint8_t)extract<uint8_t>(h), (uint8_t)extract<uint8_t>(s), (uint8_t)extract<uint8_t>(l), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t r, g, b;
        bob::ip::hsl_to_rgb_one((uint16_t)extract<uint16_t>(h), (uint16_t)extract<uint16_t>(s), (uint16_t)extract<uint16_t>(l), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_float64:
      {
        double r, g, b;
        bob::ip::hsl_to_rgb_one((double)extract<double>(h), (double)extract<double>(s), (double)extract<double>(l), r, g, b);
        return make_tuple(r, g, b);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}


// RGB to YUV and vice versa
static object rgb_to_yuv(const object& r, const object& g, const object& b, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t y, u, v;
        bob::ip::rgb_to_yuv_one((uint8_t)extract<uint8_t>(r), (uint8_t)extract<uint8_t>(g), (uint8_t)extract<uint8_t>(b), y, u, v);
        return make_tuple(y, u, v);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t y, u, v;
        bob::ip::rgb_to_yuv_one((uint16_t)extract<uint16_t>(r), (uint16_t)extract<uint16_t>(g), (uint16_t)extract<uint16_t>(b), y, u, v);
        return make_tuple(y, u, v);
      }
    case bob::core::array::t_float64:
      {
        double y, u, v;
        bob::ip::rgb_to_yuv_one((double)extract<double>(r), (double)extract<double>(g), (double)extract<double>(b), y, u, v);
        return make_tuple(y, u, v);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}

static object yuv_to_rgb(const object& y, const object& u, const object& v, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t r, g, b;
        bob::ip::yuv_to_rgb_one((uint8_t)extract<uint8_t>(y), (uint8_t)extract<uint8_t>(u), (uint8_t)extract<uint8_t>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t r, g, b;
        bob::ip::yuv_to_rgb_one((uint16_t)extract<uint16_t>(y), (uint16_t)extract<uint16_t>(u), (uint16_t)extract<uint16_t>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_float64:
      {
        double r, g, b;
        bob::ip::yuv_to_rgb_one((double)extract<double>(y), (double)extract<double>(u), (double)extract<double>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}


//a few methods to return a dynamically allocated converted object
static
void py_rgb_to_hsv (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::rgb_to_hsv(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::rgb_to_hsv(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::rgb_to_hsv(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_rgb_to_hsv2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_rgb_to_hsv(from, to);
  return to.self();
}

static
void py_hsv_to_rgb (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::hsv_to_rgb(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::hsv_to_rgb(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::hsv_to_rgb(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_hsv_to_rgb2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_hsv_to_rgb(from, to);
  return to.self();
}

static
void py_rgb_to_hsl (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::rgb_to_hsl(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::rgb_to_hsl(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::rgb_to_hsl(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_rgb_to_hsl2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_rgb_to_hsl(from, to);
  return to.self();
}

static
void py_hsl_to_rgb (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::hsl_to_rgb(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::hsl_to_rgb(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::hsl_to_rgb(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_hsl_to_rgb2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_hsl_to_rgb(from, to);
  return to.self();
}

static
void py_rgb_to_yuv (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::rgb_to_yuv(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::rgb_to_yuv(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::rgb_to_yuv(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_rgb_to_yuv2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_rgb_to_yuv(from, to);
  return to.self();
}

static
void py_yuv_to_rgb (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::yuv_to_rgb(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::yuv_to_rgb(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::yuv_to_rgb(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_yuv_to_rgb2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_yuv_to_rgb(from, to);
  return to.self();
}

static
void py_rgb_to_gray (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,2> to_ = to.bz<uint8_t,2>();
        bob::ip::rgb_to_gray(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,2> to_ = to.bz<uint16_t,2>();
        bob::ip::rgb_to_gray(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,2> to_ = to.bz<double,2>();
        bob::ip::rgb_to_gray(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_rgb_to_gray2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  if (info.nd != 3) {
    PYTHON_ERROR(TypeError,
      "input type must have at least 3 dimensions, but you gave me '%s'",
      info.str().c_str());
  }
  bob::python::ndarray to(info.dtype, info.shape[1], info.shape[2]);
  py_rgb_to_gray(from, to);
  return to.self();
}


static
void py_gray_to_rgb (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::gray_to_rgb(from.bz<uint8_t,2>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::gray_to_rgb(from.bz<uint16_t,2>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::gray_to_rgb(from.bz<double,2>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_gray_to_rgb2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info.dtype, (size_t)3, info.shape[0], info.shape[1]);
  py_gray_to_rgb(from, to);
  return to.self();
}


static const char* rgb_to_hsv_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with HSV equivalents as determined by rgb_to_hsv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* hsv_to_rgb_doc = "Takes a 3-dimensional array encoded as HSV and sets the second array with RGB equivalents as determined by hsv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* rgb_to_hsl_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with HSL equivalents as determined by rgb_to_hsl_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* hsl_to_rgb_doc = "Takes a 3-dimensional array encoded as HSL and sets the second array with RGB equivalents as determined by hsl_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* rgb_to_yuv_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with YUV (Y'CbCr) equivalents as determined by rgb_to_yuv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* yuv_to_rgb_doc = "Takes a 3-dimensional array encoded as YUV (Y'CbCr) and sets the second array with RGB equivalents as determined by yuv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* rgb_to_gray_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with gray equivalents as determined by rgb_to_gray_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array is a 2D array with the same element type. The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported";
static const char* gray_to_rgb_doc = "Takes a 2-dimensional array encoded as grays and sets the second array with RGB equivalents as determined by gray_to_rgb_one(). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported";

void bind_ip_color()
{
  // Single pixel conversions
  def("rgb_to_hsv", &rgb_to_hsv, (arg("red"), arg("green"), arg("blue"), arg("dtype")), "Converts a RGB color-pixel to HSV as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,v) values.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("hsv_to_rgb", &hsv_to_rgb, (arg("hue"), arg("saturation"), arg("value"), arg("dtype")), "Converts a HSV color-pixel to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("rgb_to_hsl", &rgb_to_hsl, (arg("red"), arg("green"), arg("blue"), arg("dtype")), "Converts a RGB color-pixel to HSL as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,l) values.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("hsl_to_rgb", &hsl_to_rgb, (arg("hue"), arg("saturation"), arg("lightness"), arg("dtype")), "Converts a HSL color-pixel to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("rgb_to_yuv", &rgb_to_yuv, (arg("red"), arg("green"), arg("blue"), arg("dtype")), "Converts a RGB color-coded pixel to YUV (Y'CbCr) using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values.\n\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("yuv_to_rgb", &yuv_to_rgb, (arg("y"), arg("u"), arg("v"), arg("dtype")), "Converts a YUV (Y'CbCr) color-coded pixel to RGB using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.\n\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("rgb_to_gray", &rgb_to_gray, (arg("red"), arg("green"), arg("blue"), arg("dtype")), "Converts a RGB color-coded pixel to Grayscale using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component) in the desired data format. This method is more efficient than calling rgb_to_yuv() method just to extract the Y component.\n\n Depending on the dtype parameter, different types of data is expected:\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("gray_to_rgb", &gray_to_rgb, (arg("y"), arg("dtype")), "Converts a grayscale pixel to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");

  // image conversions from source to target image
  def("rgb_to_hsv", &py_rgb_to_hsv, (arg("rgb"), arg("hsv")), rgb_to_hsv_doc);
  def("hsv_to_rgb", &py_hsv_to_rgb, (arg("hsv"), arg("rgb")), hsv_to_rgb_doc);
  def("rgb_to_hsl", &py_rgb_to_hsl, (arg("rgb"), arg("hsl")), rgb_to_hsl_doc);
  def("hsl_to_rgb", &py_hsl_to_rgb, (arg("hsl"), arg("rgb")), hsl_to_rgb_doc);
  def("rgb_to_yuv", &py_rgb_to_yuv, (arg("rgb"), arg("yuv")), rgb_to_yuv_doc);
  def("yuv_to_rgb", &py_yuv_to_rgb, (arg("yuv"), arg("rgb")), yuv_to_rgb_doc);
  def("rgb_to_gray", &py_rgb_to_gray, (arg("rgb"), arg("gray")), rgb_to_gray_doc);
  def("gray_to_rgb", &py_gray_to_rgb, (arg("gray"), arg("rgb")), gray_to_rgb_doc);

  // more pythonic versions that return a dynamically allocated result
  def("rgb_to_hsv", &py_rgb_to_hsv2, (arg("rgb")), rgb_to_hsv_doc);
  def("hsv_to_rgb", &py_hsv_to_rgb2, (arg("hsv")), hsv_to_rgb_doc);
  def("rgb_to_hsl", &py_rgb_to_hsl2, (arg("rgb")), rgb_to_hsl_doc);
  def("hsl_to_rgb", &py_hsl_to_rgb2, (arg("hsl")), hsl_to_rgb_doc);
  def("rgb_to_yuv", &py_rgb_to_yuv2, (arg("rgb")), rgb_to_yuv_doc);
  def("yuv_to_rgb", &py_yuv_to_rgb2, (arg("yuv")), yuv_to_rgb_doc);
  def("rgb_to_gray", &py_rgb_to_gray2, (arg("rgb")), rgb_to_gray_doc);
  def("gray_to_rgb", &py_gray_to_rgb2, (arg("gray")), gray_to_rgb_doc);
}
