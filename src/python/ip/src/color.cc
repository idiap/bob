/**
 * @file python/ip/src/color.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds color converters to python
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

#include "ip/color.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

template <typename T> static tuple rgb_to_hsv_one_python(T r, T g, T b) {
  T h, s, v;
  ip::rgb_to_hsv_one(r, g, b, h, s, v);
  return make_tuple(h, s, v);
}

template <> tuple rgb_to_hsv_one_python(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t h, s, v;
  ip::rgb_to_hsv_one(r, g, b, h, s, v);
  return make_tuple((uint32_t)h, (uint32_t)s, (uint32_t)v);
}

template <typename T> static tuple hsv_to_rgb_one_python(T h, T s, T v) {
  T r, g, b;
  ip::hsv_to_rgb_one(h, s, v, r, g, b);
  return make_tuple(r, g, b);
}

template <> tuple hsv_to_rgb_one_python(uint8_t h, uint8_t s, uint8_t v) {
  uint8_t r, g, b;
  ip::hsv_to_rgb_one(h, s, v, r, g, b);
  return make_tuple((uint32_t)r, (uint32_t)g, (uint32_t)b);
}

template <typename T> static tuple rgb_to_hsl_one_python(T r, T g, T b) {
  T h, s, l;
  ip::rgb_to_hsl_one(r, g, b, h, s, l);
  return make_tuple(h, s, l);
}

template <> tuple rgb_to_hsl_one_python(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t h, s, l;
  ip::rgb_to_hsl_one(r, g, b, h, s, l);
  return make_tuple((uint32_t)h, (uint32_t)s, (uint32_t)l);
}

template <typename T> static tuple hsl_to_rgb_one_python(T h, T s, T l) {
  T r, g, b;
  ip::hsl_to_rgb_one(h, s, l, r, g, b);
  return make_tuple(r, g, b);
}

template <> tuple hsl_to_rgb_one_python(uint8_t h, uint8_t s, uint8_t l) {
  uint8_t r, g, b;
  ip::hsl_to_rgb_one(h, s, l, r, g, b);
  return make_tuple((uint32_t)r, (uint32_t)g, (uint32_t)b);
}

template <typename T> static tuple rgb_to_yuv_one_python(T r, T g, T b) {
  T y, u, v;
  ip::rgb_to_yuv_one(r, g, b, y, u, v);
  return make_tuple(y, u, v);
}

template <> tuple rgb_to_yuv_one_python(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t y, u, v;
  ip::rgb_to_yuv_one(r, g, b, y, u, v);
  return make_tuple((uint32_t)y, (uint32_t)u, (uint32_t)v);
}

template <typename T> static tuple yuv_to_rgb_one_python(T y, T u, T v) {
  T r, g, b;
  ip::yuv_to_rgb_one(y, u, v, r, g, b);
  return make_tuple(r, g, b);
}

template <> tuple yuv_to_rgb_one_python(uint8_t y, uint8_t u, uint8_t v) {
  uint8_t r, g, b;
  ip::yuv_to_rgb_one(y, u, v, r, g, b);
  return make_tuple((uint32_t)r, (uint32_t)g, (uint32_t)b);
}

template <typename T> static object rgb_to_gray_one_python(T r, T g, T b) {
  T y;
  ip::rgb_to_gray_one(r, g, b, y);
  return object(y);
}

template <> object rgb_to_gray_one_python(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t y;
  ip::rgb_to_gray_one(r, g, b, y);
  return object((uint32_t)y);
}

template <typename T> static tuple gray_to_rgb_one_python(T y) {
  T r, g, b;
  ip::gray_to_rgb_one(y, r, g, b);
  return make_tuple(r, g, b);
}

template <> tuple gray_to_rgb_one_python(uint8_t y) {
  uint8_t r, g, b;
  ip::gray_to_rgb_one(y, r, g, b);
  return make_tuple((uint32_t)r, (uint32_t)g, (uint32_t)b);
}

//a few methods to return a dynamically allocated converted object
static void py_rgb_to_hsv (tp::const_ndarray from, tp::ndarray to) {
  switch (from.type().dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        ip::rgb_to_hsv(from.bz<uint8_t,3>(), to_);
      }
      break;
    case ca::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        ip::rgb_to_hsv(from.bz<uint16_t,3>(), to_);
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        ip::rgb_to_hsv(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "color conversion operator does not support array with type '%s'", from.type().str().c_str());
  }
}

static object py_rgb_to_hsv2 (tp::const_ndarray from) {
  const ca::typeinfo& info = from.type();
  tp::ndarray to(info);
  py_rgb_to_hsv(from, to);
  return to.self();
}

static void py_hsv_to_rgb (tp::const_ndarray from, tp::ndarray to) {
  switch (from.type().dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        ip::hsv_to_rgb(from.bz<uint8_t,3>(), to_);
      }
      break;
    case ca::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        ip::hsv_to_rgb(from.bz<uint16_t,3>(), to_);
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        ip::hsv_to_rgb(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "color conversion operator does not support array with type '%s'", from.type().str().c_str());
  }
}

static object py_hsv_to_rgb2 (tp::const_ndarray from) {
  const ca::typeinfo& info = from.type();
  tp::ndarray to(info);
  py_hsv_to_rgb(from, to);
  return to.self();
}

static void py_rgb_to_hsl (tp::const_ndarray from, tp::ndarray to) {
  switch (from.type().dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        ip::rgb_to_hsl(from.bz<uint8_t,3>(), to_);
      }
      break;
    case ca::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        ip::rgb_to_hsl(from.bz<uint16_t,3>(), to_);
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        ip::rgb_to_hsl(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "color conversion operator does not support array with type '%s'", from.type().str().c_str());
  }
}

static object py_rgb_to_hsl2 (tp::const_ndarray from) {
  const ca::typeinfo& info = from.type();
  tp::ndarray to(info);
  py_rgb_to_hsl(from, to);
  return to.self();
}

static void py_hsl_to_rgb (tp::const_ndarray from, tp::ndarray to) {
  switch (from.type().dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        ip::hsl_to_rgb(from.bz<uint8_t,3>(), to_);
      }
      break;
    case ca::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        ip::hsl_to_rgb(from.bz<uint16_t,3>(), to_);
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        ip::hsl_to_rgb(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "color conversion operator does not support array with type '%s'", from.type().str().c_str());
  }
}

static object py_hsl_to_rgb2 (tp::const_ndarray from) {
  const ca::typeinfo& info = from.type();
  tp::ndarray to(info);
  py_hsl_to_rgb(from, to);
  return to.self();
}

static void py_rgb_to_yuv (tp::const_ndarray from, tp::ndarray to) {
  switch (from.type().dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        ip::rgb_to_yuv(from.bz<uint8_t,3>(), to_);
      }
      break;
    case ca::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        ip::rgb_to_yuv(from.bz<uint16_t,3>(), to_);
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        ip::rgb_to_yuv(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "color conversion operator does not support array with type '%s'", from.type().str().c_str());
  }
}

static object py_rgb_to_yuv2 (tp::const_ndarray from) {
  const ca::typeinfo& info = from.type();
  tp::ndarray to(info);
  py_rgb_to_yuv(from, to);
  return to.self();
}

static void py_yuv_to_rgb (tp::const_ndarray from, tp::ndarray to) {
  switch (from.type().dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        ip::yuv_to_rgb(from.bz<uint8_t,3>(), to_);
      }
      break;
    case ca::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        ip::yuv_to_rgb(from.bz<uint16_t,3>(), to_);
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        ip::yuv_to_rgb(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "color conversion operator does not support array with type '%s'", from.type().str().c_str());
  }
}

static object py_yuv_to_rgb2 (tp::const_ndarray from) {
  const ca::typeinfo& info = from.type();
  tp::ndarray to(info);
  py_yuv_to_rgb(from, to);
  return to.self();
}

static void py_rgb_to_gray (tp::const_ndarray from, tp::ndarray to) {
  switch (from.type().dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,2> to_ = to.bz<uint8_t,2>();
        ip::rgb_to_gray(from.bz<uint8_t,3>(), to_);
      }
      break;
    case ca::t_uint16:
      {
        blitz::Array<uint16_t,2> to_ = to.bz<uint16_t,2>();
        ip::rgb_to_gray(from.bz<uint16_t,3>(), to_);
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,2> to_ = to.bz<double,2>();
        ip::rgb_to_gray(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "color conversion operator does not support array with type '%s'", from.type().str().c_str());
  }
}

static object py_rgb_to_gray2 (tp::const_ndarray from) {
  const ca::typeinfo& info = from.type();
  if (info.nd != 3) {
    PYTHON_ERROR(TypeError, "input type must have at least 3 dimensions, but you gave me '%s'", info.str().c_str());
  }
  tp::ndarray to(info.dtype, info.shape[1], info.shape[2]);
  py_rgb_to_gray(from, to);
  return to.self();
}


static void py_gray_to_rgb (tp::const_ndarray from, tp::ndarray to) {
  switch (from.type().dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        ip::gray_to_rgb(from.bz<uint8_t,2>(), to_);
      }
      break;
    case ca::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        ip::gray_to_rgb(from.bz<uint16_t,2>(), to_);
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        ip::gray_to_rgb(from.bz<double,2>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "color conversion operator does not support array with type '%s'", from.type().str().c_str());
  }
}

static object py_gray_to_rgb2 (tp::const_ndarray from) {
  const ca::typeinfo& info = from.type();
  tp::ndarray to(info.dtype, (size_t)3, info.shape[0], info.shape[1]);
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
  //Single pixel conversions
  def("rgb_to_hsv_u8", &rgb_to_hsv_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band with 256 gray levels) to HSV as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,v) values.");
  def("rgb_to_hsv_u16", &rgb_to_hsv_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band with 65535 gray levels) to HSV as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,v) values.");
  def("rgb_to_hsv_f", &rgb_to_hsv_one_python<double>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band using a float between 0 and 1) to HSV as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,v) values.");
  def("hsv_to_rgb_u8", &hsv_to_rgb_one_python<uint8_t>, (arg("hue"), arg("saturation"), arg("value")), "Converts a HSV color-pixel (each band with 256 gray levels) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("hsv_to_rgb_u16", &hsv_to_rgb_one_python<uint16_t>, (arg("hue"), arg("saturation"), arg("value")), "Converts a HSV color-pixel (each band with 65535 gray levels) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("hsv_to_rgb_f", &hsv_to_rgb_one_python<double>, (arg("hue"), arg("saturation"), arg("value")), "Converts a HSV color-pixel (each band using a float between 0 and 1) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("rgb_to_hsl_u8", &rgb_to_hsl_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band with 256 gray levels) to HSL as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,l) values.");
  def("rgb_to_hsl_u16", &rgb_to_hsl_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band with 65535 gray levels) to HSL as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,l) values.");
  def("rgb_to_hsl_f", &rgb_to_hsl_one_python<double>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band using a float between 0 and 1) to HSL as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,l) values.");
  def("hsl_to_rgb_u8", &hsl_to_rgb_one_python<uint8_t>, (arg("hue"), arg("saturation"), arg("lightness")), "Converts a HSL color-pixel (each band with 256 gray levels) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("hsl_to_rgb_u16", &hsl_to_rgb_one_python<uint16_t>, (arg("hue"), arg("saturation"), arg("lightness")), "Converts a HSL color-pixel (each band with 65535 gray levels) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("hsl_to_rgb_f", &hsl_to_rgb_one_python<double>, (arg("hue"), arg("saturation"), arg("lightness")), "Converts a HSL color-pixel (each band using a float between 0 and 1) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("rgb_to_yuv_u8", &rgb_to_yuv_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 256 levels of gray) to YUV (Y'CbCr) using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values.");
  def("rgb_to_yuv_u16", &rgb_to_yuv_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 65535 levels of gray) to YUV (Y'CbCr) using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values.");
  def("rgb_to_yuv_f", &rgb_to_yuv_one_python<double>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands of floats between 0 and 1) to YUV (Y'CbCr) using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values. WARNING: This implementation returns U and V values varying from 0 to 1 for mapping norm ranges [-0.5, 0.5] into a more standard setting.");
  def("yuv_to_rgb_u8", &yuv_to_rgb_one_python<uint8_t>, (arg("y"), arg("u"), arg("v")), "Converts a YUV (Y'CbCr) color-coded pixel (3-bands each with 256 levels of gray) to RGB using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.");
  def("yuv_to_rgb_u16", &yuv_to_rgb_one_python<uint16_t>, (arg("y"), arg("u"), arg("v")), "Converts a YUV (Y'CbCr) color-coded pixel (3-bands each with 65535 levels of gray) to RGB using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.");
  def("yuv_to_rgb_f", &yuv_to_rgb_one_python<double>, (arg("y"), arg("u"), arg("v")), "Converts a YUV (Y'CbCr) color-coded pixel (3-bands of floats between 0 and 1) to RGB using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.");
  def("rgb_to_gray_u8", &rgb_to_gray_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 256 levels of gray) to Grayscale using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component). This method is more efficient than calling rgb_to_yuv*() methods just to extract the Y component. Returns the grayscale value quantized to 256 levels of gray.");
  def("rgb_to_gray_u16", &rgb_to_gray_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 65535 levels of gray) to Grayscale using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component). This method is more efficient than calling rgb_to_yuv*() methods just to extract the Y component. Returns the grayscale value quantized to 256 levels of gray.");
  def("rgb_to_gray_f", &rgb_to_gray_one_python<double>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each using a float between 0 and 1) to Grayscale using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component). This method is more efficient than calling rgb_to_yuv*() methods just to extract the Y component. Returns the grayscale value quantized to 256 levels of gray.");
  def("gray_to_rgb_u8", &gray_to_rgb_one_python<uint8_t>, (arg("y")), "Converts a grayscale color-coded pixel (with 256 levels of gray) to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.");
  def("gray_to_rgb_u16", &gray_to_rgb_one_python<uint16_t>, (arg("y")), "Converts a grayscale color-coded pixel (with 65535 levels of gray) to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.");
  def("gray_to_rgb_f", &gray_to_rgb_one_python<double>, (arg("y")), "Converts a grayscale color-coded pixel (float between 0 and 1) to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.");

  //more pythonic versions that return a dynamically allocated result
  def("rgb_to_hsv", &py_rgb_to_hsv, (arg("rgb"), arg("hsv")), rgb_to_hsv_doc);
  def("hsv_to_rgb", &py_hsv_to_rgb, (arg("hsv"), arg("rgb")), hsv_to_rgb_doc);
  def("rgb_to_hsl", &py_rgb_to_hsl, (arg("rgb"), arg("hsl")), rgb_to_hsl_doc);
  def("hsl_to_rgb", &py_hsl_to_rgb, (arg("hsl"), arg("rgb")), hsl_to_rgb_doc);
  def("rgb_to_yuv", &py_rgb_to_yuv, (arg("rgb"), arg("yuv")), rgb_to_yuv_doc);
  def("yuv_to_rgb", &py_yuv_to_rgb, (arg("yuv"), arg("rgb")), yuv_to_rgb_doc);
  def("rgb_to_gray", &py_rgb_to_gray, (arg("rgb"), arg("gray")), rgb_to_gray_doc);
  def("gray_to_rgb", &py_gray_to_rgb, (arg("gray"), arg("rgb")), gray_to_rgb_doc);

  def("rgb_to_hsv", &py_rgb_to_hsv2, (arg("rgb")), rgb_to_hsv_doc);
  def("hsv_to_rgb", &py_hsv_to_rgb2, (arg("hsv")), hsv_to_rgb_doc);
  def("rgb_to_hsl", &py_rgb_to_hsl2, (arg("rgb")), rgb_to_hsl_doc);
  def("hsl_to_rgb", &py_hsl_to_rgb2, (arg("hsl")), hsl_to_rgb_doc);
  def("rgb_to_yuv", &py_rgb_to_yuv2, (arg("rgb")), rgb_to_yuv_doc);
  def("yuv_to_rgb", &py_yuv_to_rgb2, (arg("yuv")), yuv_to_rgb_doc);
  def("rgb_to_gray", &py_rgb_to_gray2, (arg("rgb")), rgb_to_gray_doc);
  def("gray_to_rgb", &py_gray_to_rgb2, (arg("gray")), gray_to_rgb_doc);
}
