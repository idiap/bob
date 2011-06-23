/**
 * @file python/ip/src/color.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds color converters to python 
 */

#include <boost/python.hpp>

#include "ip/color.h"
#include "core/python/exception.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tpy = Torch::core::python;

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
template <typename T> blitz::Array<T,3> py_rgb_to_hsv 
  (const blitz::Array<T,3>& from) {
    blitz::Array<T,3> to(from.shape());
    ip::rgb_to_hsv(from, to);
    return to;
}

template <typename T> blitz::Array<T,3> py_hsv_to_rgb
  (const blitz::Array<T,3>& from) {
    blitz::Array<T,3> to(from.shape());
    ip::hsv_to_rgb(from, to);
    return to;
}

template <typename T> blitz::Array<T,3> py_rgb_to_hsl 
  (const blitz::Array<T,3>& from) {
    blitz::Array<T,3> to(from.shape());
    ip::rgb_to_hsl(from, to);
    return to;
}

template <typename T> blitz::Array<T,3> py_hsl_to_rgb
  (const blitz::Array<T,3>& from) {
    blitz::Array<T,3> to(from.shape());
    ip::hsl_to_rgb(from, to);
    return to;
}

template <typename T> blitz::Array<T,3> py_rgb_to_yuv
  (const blitz::Array<T,3>& from) {
    blitz::Array<T,3> to(from.shape());
    ip::rgb_to_yuv(from, to);
    return to;
}

template <typename T> blitz::Array<T,3> py_yuv_to_rgb
  (const blitz::Array<T,3>& from) {
    blitz::Array<T,3> to(from.shape());
    ip::yuv_to_rgb(from, to);
    return to;
}

template <typename T> blitz::Array<T,2> py_rgb_to_gray
  (const blitz::Array<T,3>& from) {
    blitz::Array<T,2> to(from.extent(1), from.extent(2));
    ip::rgb_to_gray(from, to);
    return to;
}

template <typename T> blitz::Array<T,3> py_gray_to_rgb
  (const blitz::Array<T,2>& from) {
    blitz::Array<T,3> to(3, from.extent(0), from.extent(1));
    ip::gray_to_rgb(from, to);
    return to;
}

template <typename T> static void bind_type() {
  static const char* rgb_to_hsv_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with HSV equivalents as determined by rgb_to_hsv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
  static const char* hsv_to_rgb_doc = "Takes a 3-dimensional array encoded as HSV and sets the second array with RGB equivalents as determined by hsv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
  static const char* rgb_to_hsl_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with HSL equivalents as determined by rgb_to_hsl_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
  static const char* hsl_to_rgb_doc = "Takes a 3-dimensional array encoded as HSL and sets the second array with RGB equivalents as determined by hsl_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
  static const char* rgb_to_yuv_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with YUV (Y'CbCr) equivalents as determined by rgb_to_yuv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
  static const char* yuv_to_rgb_doc = "Takes a 3-dimensional array encoded as YUV (Y'CbCr) and sets the second array with RGB equivalents as determined by yuv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
  static const char* rgb_to_gray_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with gray equivalents as determined by rgb_to_gray_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array is a 2D array with the same element type. The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported";
  static const char* gray_to_rgb_doc = "Takes a 2-dimensional array encoded as grays and sets the second array with RGB equivalents as determined by gray_to_rgb_one(). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported";

  def("rgb_to_hsv", &ip::rgb_to_hsv<T>, (arg("rgb"), arg("hsv")), rgb_to_hsv_doc);
  def("hsv_to_rgb", &ip::hsv_to_rgb<T>, (arg("hsv"), arg("rgb")), hsv_to_rgb_doc);
  def("rgb_to_hsl", &ip::rgb_to_hsl<T>, (arg("rgb"), arg("hsl")), rgb_to_hsl_doc);
  def("hsl_to_rgb", &ip::hsl_to_rgb<T>, (arg("hsl"), arg("rgb")), hsl_to_rgb_doc);
  def("rgb_to_yuv", &ip::rgb_to_yuv<T>, (arg("rgb"), arg("yuv")), rgb_to_yuv_doc);
  def("yuv_to_rgb", &ip::yuv_to_rgb<T>, (arg("yuv"), arg("rgb")), yuv_to_rgb_doc);
  def("rgb_to_gray", &ip::rgb_to_gray<T>, (arg("rgb"), arg("gray")), rgb_to_gray_doc);
  def("gray_to_rgb", &ip::gray_to_rgb<T>, (arg("gray"), arg("rgb")), gray_to_rgb_doc);

  //more pythonic versions that return a dynamically allocated result
  def("rgb_to_hsv", &py_rgb_to_hsv<T>, (arg("rgb")), rgb_to_hsv_doc);
  def("hsv_to_rgb", &py_hsv_to_rgb<T>, (arg("hsv")), hsv_to_rgb_doc);
  def("rgb_to_hsl", &py_rgb_to_hsl<T>, (arg("rgb")), rgb_to_hsl_doc);
  def("hsl_to_rgb", &py_hsl_to_rgb<T>, (arg("hsl")), hsl_to_rgb_doc);
  def("rgb_to_yuv", &py_rgb_to_yuv<T>, (arg("rgb")), rgb_to_yuv_doc);
  def("yuv_to_rgb", &py_yuv_to_rgb<T>, (arg("yuv")), yuv_to_rgb_doc);
  def("rgb_to_gray", &py_rgb_to_gray<T>, (arg("rgb")), rgb_to_gray_doc);
  def("gray_to_rgb", &py_gray_to_rgb<T>, (arg("gray")), gray_to_rgb_doc);
}

void bind_ip_color()
{
  //Exceptions for this functionality

  tpy::CxxToPythonTranslatorPar<Torch::ip::UnsupportedTypeForColorConversion, Torch::ip::Exception, Torch::core::array::ElementType>("UnsupportedTypeForColorConversion", "This exception is thrown when the color conversion for a particular type is not implemented in torch");

  tpy::CxxToPythonTranslatorPar2<Torch::ip::UnsupportedRowExtent, Torch::ip::Exception, int, int>("UnsupportedRowExtent", "This exception is thrown when the input matrix does not conform to the method specifications in number of rows.");
#include "core/python/exception.h"

  //Single pixel conversions
  
  def("rgb_to_hsv_u8", &rgb_to_hsv_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band with 256 gray levels) to HSV as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,v) values.");
  def("rgb_to_hsv_u16", &rgb_to_hsv_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band with 65535 gray levels) to HSV as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,v) values.");
  def("rgb_to_hsv_f", &rgb_to_hsv_one_python<float>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band using a float between 0 and 1) to HSV as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,v) values.");
  def("hsv_to_rgb_u8", &hsv_to_rgb_one_python<uint8_t>, (arg("hue"), arg("saturation"), arg("value")), "Converts a HSV color-pixel (each band with 256 gray levels) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("hsv_to_rgb_u16", &hsv_to_rgb_one_python<uint16_t>, (arg("hue"), arg("saturation"), arg("value")), "Converts a HSV color-pixel (each band with 65535 gray levels) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("hsv_to_rgb_f", &hsv_to_rgb_one_python<float>, (arg("hue"), arg("saturation"), arg("value")), "Converts a HSV color-pixel (each band using a float between 0 and 1) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("rgb_to_hsl_u8", &rgb_to_hsl_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band with 256 gray levels) to HSL as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,l) values.");
  def("rgb_to_hsl_u16", &rgb_to_hsl_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band with 65535 gray levels) to HSL as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,l) values.");
  def("rgb_to_hsl_f", &rgb_to_hsl_one_python<float>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-pixel (each band using a float between 0 and 1) to HSL as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,l) values.");
  def("hsl_to_rgb_u8", &hsl_to_rgb_one_python<uint8_t>, (arg("hue"), arg("saturation"), arg("lightness")), "Converts a HSL color-pixel (each band with 256 gray levels) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("hsl_to_rgb_u16", &hsl_to_rgb_one_python<uint16_t>, (arg("hue"), arg("saturation"), arg("lightness")), "Converts a HSL color-pixel (each band with 65535 gray levels) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("hsl_to_rgb_f", &hsl_to_rgb_one_python<float>, (arg("hue"), arg("saturation"), arg("lightness")), "Converts a HSL color-pixel (each band using a float between 0 and 1) to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.");
  def("rgb_to_yuv_u8", &rgb_to_yuv_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 256 levels of gray) to YUV (Y'CbCr) using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values.");
  def("rgb_to_yuv_u16", &rgb_to_yuv_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 65535 levels of gray) to YUV (Y'CbCr) using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values.");
  def("rgb_to_yuv_f", &rgb_to_yuv_one_python<float>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands of floats between 0 and 1) to YUV (Y'CbCr) using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values. WARNING: This implementation returns U and V values varying from 0 to 1 for mapping norm ranges [-0.5, 0.5] into a more standard setting.");
  def("yuv_to_rgb_u8", &yuv_to_rgb_one_python<uint8_t>, (arg("y"), arg("u"), arg("v")), "Converts a YUV (Y'CbCr) color-coded pixel (3-bands each with 256 levels of gray) to RGB using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.");
  def("yuv_to_rgb_u16", &yuv_to_rgb_one_python<uint16_t>, (arg("y"), arg("u"), arg("v")), "Converts a YUV (Y'CbCr) color-coded pixel (3-bands each with 65535 levels of gray) to RGB using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.");
  def("yuv_to_rgb_f", &yuv_to_rgb_one_python<float>, (arg("y"), arg("u"), arg("v")), "Converts a YUV (Y'CbCr) color-coded pixel (3-bands of floats between 0 and 1) to RGB using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.");
  def("rgb_to_gray_u8", &rgb_to_gray_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 256 levels of gray) to Grayscale using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component). This method is more efficient than calling rgb_to_yuv*() methods just to extract the Y component. Returns the grayscale value quantized to 256 levels of gray.");
  def("rgb_to_gray_u16", &rgb_to_gray_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 65535 levels of gray) to Grayscale using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component). This method is more efficient than calling rgb_to_yuv*() methods just to extract the Y component. Returns the grayscale value quantized to 256 levels of gray.");
  def("rgb_to_gray_f", &rgb_to_gray_one_python<float>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each using a float between 0 and 1) to Grayscale using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component). This method is more efficient than calling rgb_to_yuv*() methods just to extract the Y component. Returns the grayscale value quantized to 256 levels of gray.");
  def("gray_to_rgb_u8", &gray_to_rgb_one_python<uint8_t>, (arg("y")), "Converts a grayscale color-coded pixel (with 256 levels of gray) to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.");
  def("gray_to_rgb_u16", &gray_to_rgb_one_python<uint16_t>, (arg("y")), "Converts a grayscale color-coded pixel (with 65535 levels of gray) to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.");
  def("gray_to_rgb_f", &gray_to_rgb_one_python<float>, (arg("y")), "Converts a grayscale color-coded pixel (float between 0 and 1) to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.");

  //Full matrix conversions
  bind_type<uint8_t>();
  bind_type<uint16_t>();
  bind_type<float>();

}
