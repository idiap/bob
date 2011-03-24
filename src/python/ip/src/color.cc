/**
 * @file python/ip/src/color.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds color converters to python 
 */

#include <boost/python.hpp>

#include "ip/color.h"

using namespace boost::python;
namespace ip = Torch::ip;

template <typename T>
static tuple rgb_to_hsv_one_python(T r, T g, T b) {
  T h, s, v;
  ip::rgb_to_hsv_one(r, g, b, h, s, v);
  return make_tuple(h, s, v);
}

template <>
static tuple rgb_to_hsv_one_python(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t h, s, v;
  ip::rgb_to_hsv_one(r, g, b, h, s, v);
  return make_tuple((uint32_t)h, (uint32_t)s, (uint32_t)v);
}

template <typename T>
static tuple hsv_to_rgb_one_python(T h, T s, T v) {
  T r, g, b;
  ip::hsv_to_rgb_one(h, s, v, r, g, b);
  return make_tuple(r, g, b);
}

template <>
static tuple hsv_to_rgb_one_python(uint8_t h, uint8_t s, uint8_t v) {
  uint8_t r, g, b;
  ip::hsv_to_rgb_one(h, s, v, r, g, b);
  return make_tuple((uint32_t)r, (uint32_t)g, (uint32_t)b);
}

template <typename T>
static tuple rgb_to_hsl_one_python(T r, T g, T b) {
  T h, s, l;
  ip::rgb_to_hsl_one(r, g, b, h, s, l);
  return make_tuple(h, s, l);
}

template <>
static tuple rgb_to_hsl_one_python(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t h, s, l;
  ip::rgb_to_hsl_one(r, g, b, h, s, l);
  return make_tuple((uint32_t)h, (uint32_t)s, (uint32_t)l);
}

template <typename T>
static tuple hsl_to_rgb_one_python(T h, T s, T l) {
  T r, g, b;
  ip::hsl_to_rgb_one(h, s, l, r, g, b);
  return make_tuple(r, g, b);
}

template <>
static tuple hsl_to_rgb_one_python(uint8_t h, uint8_t s, uint8_t l) {
  uint8_t r, g, b;
  ip::hsl_to_rgb_one(h, s, l, r, g, b);
  return make_tuple((uint32_t)r, (uint32_t)g, (uint32_t)b);
}

template <typename T>
static tuple rgb_to_yuv_one_python(T r, T g, T b) {
  T y, u, v;
  ip::rgb_to_yuv_one(r, g, b, y, u, v);
  return make_tuple(y, u, v);
}

template <>
static tuple rgb_to_yuv_one_python(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t y, u, v;
  ip::rgb_to_yuv_one(r, g, b, y, u, v);
  return make_tuple((uint32_t)y, (uint32_t)u, (uint32_t)v);
}

template <typename T>
static tuple yuv_to_rgb_one_python(T y, T u, T v) {
  T r, g, b;
  ip::yuv_to_rgb_one(y, u, v, r, g, b);
  return make_tuple(r, g, b);
}

template <>
static tuple yuv_to_rgb_one_python(uint8_t y, uint8_t u, uint8_t v) {
  uint8_t r, g, b;
  ip::yuv_to_rgb_one(y, u, v, r, g, b);
  return make_tuple((uint32_t)r, (uint32_t)g, (uint32_t)b);
}

template <typename T>
static object rgb_to_gray_one_python(T r, T g, T b) {
  T y;
  ip::rgb_to_gray_one(r, g, b, y);
  return object(y);
}

template <>
static object rgb_to_gray_one_python(uint8_t r, uint8_t g, uint8_t b) {
  uint8_t y;
  ip::rgb_to_gray_one(r, g, b, y);
  return object((uint32_t)y);
}

template <typename T>
static tuple gray_to_rgb_one_python(T y) {
  T r, g, b;
  ip::gray_to_rgb_one(y, r, g, b);
  return make_tuple(r, g, b);
}

template <>
static tuple gray_to_rgb_one_python(uint8_t y) {
  uint8_t r, g, b;
  ip::gray_to_rgb_one(y, r, g, b);
  return make_tuple((uint32_t)r, (uint32_t)g, (uint32_t)b);
}

void bind_ip_color()
{
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
  def("rgb_to_yuv_u8", &rgb_to_yuv_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 256 levels of gray) to YUV (Y'CbCr) using the CCIR 601 norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values.");
  def("rgb_to_yuv_u16", &rgb_to_yuv_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 65535 levels of gray) to YUV (Y'CbCr) using the CCIR 601 norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values.");
  def("rgb_to_yuv_f", &rgb_to_yuv_one_python<float>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands of floats between 0 and 1) to YUV (Y'CbCr) using the CCIR 601 norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values.");
  def("yuv_to_rgb_u8", &yuv_to_rgb_one_python<uint8_t>, (arg("y"), arg("u"), arg("v")), "Converts a YUV (Y'CbCr) color-coded pixel (3-bands each with 256 levels of gray) to RGB using the CCIR 601 norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.");
  def("yuv_to_rgb_u16", &yuv_to_rgb_one_python<uint16_t>, (arg("y"), arg("u"), arg("v")), "Converts a YUV (Y'CbCr) color-coded pixel (3-bands each with 65535 levels of gray) to RGB using the CCIR 601 norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.");
  def("yuv_to_rgb_f", &yuv_to_rgb_one_python<float>, (arg("y"), arg("u"), arg("v")), "Converts a YUV (Y'CbCr) color-coded pixel (3-bands of floats between 0 and 1) to RGB using the CCIR 601 norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.");
  def("rgb_to_gray_u8", &rgb_to_gray_one_python<uint8_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 256 levels of gray) to Grayscale using the CCIR 601 norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component). This method is more efficient than calling rgb_to_yuv*() methods just to extract the Y component. Returns the grayscale value quantized to 256 levels of gray.");
  def("rgb_to_gray_u16", &rgb_to_gray_one_python<uint16_t>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each with 65535 levels of gray) to Grayscale using the CCIR 601 norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component). This method is more efficient than calling rgb_to_yuv*() methods just to extract the Y component. Returns the grayscale value quantized to 256 levels of gray.");
  def("rgb_to_gray_f", &rgb_to_gray_one_python<float>, (arg("red"), arg("green"), arg("blue")), "Converts a RGB color-coded pixel (3-bands each using a float between 0 and 1) to Grayscale using the CCIR 601 norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component). This method is more efficient than calling rgb_to_yuv*() methods just to extract the Y component. Returns the grayscale value quantized to 256 levels of gray.");
  def("gray_to_rgb_u8", &gray_to_rgb_one_python<uint8_t>, (arg("y")), "Converts a grayscale color-coded pixel (with 256 levels of gray) to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.");
  def("gray_to_rgb_u16", &gray_to_rgb_one_python<uint16_t>, (arg("y")), "Converts a grayscale color-coded pixel (with 65535 levels of gray) to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.");
  def("gray_to_rgb_f", &gray_to_rgb_one_python<float>, (arg("y")), "Converts a grayscale color-coded pixel (float between 0 and 1) to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.");

  //Full matrix conversions

  def("rgb_to_hsv", &ip::rgb_to_hsv<uint8_t>, (arg("rgb"), arg("hsv")), "Takes a 3-dimensional array encoded as RGB and sets the second array with HSV equivalents as determined by rgb_to_hsv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("rgb_to_hsv", &ip::rgb_to_hsv<uint16_t>, (arg("rgb"), arg("hsv")), "Takes a 3-dimensional array encoded as RGB and sets the second array with HSV equivalents as determined by rgb_to_hsv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("rgb_to_hsv", &ip::rgb_to_hsv<float>, (arg("rgb"), arg("hsv")), "Takes a 3-dimensional array encoded as RGB and sets the second array with HSV equivalents as determined by rgb_to_hsv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("hsv_to_rgb", &ip::hsv_to_rgb<uint8_t>, (arg("hsv"), arg("rgb")), "Takes a 3-dimensional array encoded as HSV and sets the second array with RGB equivalents as determined by hsv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("hsv_to_rgb", &ip::hsv_to_rgb<uint16_t>, (arg("hsv"), arg("rgb")), "Takes a 3-dimensional array encoded as HSV and sets the second array with RGB equivalents as determined by hsv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("hsv_to_rgb", &ip::hsv_to_rgb<float>, (arg("hsv"), arg("rgb")), "Takes a 3-dimensional array encoded as HSV and sets the second array with RGB equivalents as determined by hsv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("rgb_to_hsl", &ip::rgb_to_hsl<uint8_t>, (arg("rgb"), arg("hsl")), "Takes a 3-dimensional array encoded as RGB and sets the second array with HSL equivalents as determined by rgb_to_hsl_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("rgb_to_hsl", &ip::rgb_to_hsl<uint16_t>, (arg("rgb"), arg("hsl")), "Takes a 3-dimensional array encoded as RGB and sets the second array with HSL equivalents as determined by rgb_to_hsl_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("rgb_to_hsl", &ip::rgb_to_hsl<float>, (arg("rgb"), arg("hsl")), "Takes a 3-dimensional array encoded as RGB and sets the second array with HSL equivalents as determined by rgb_to_hsl_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("hsl_to_rgb", &ip::hsl_to_rgb<uint8_t>, (arg("hsl"), arg("rgb")), "Takes a 3-dimensional array encoded as HSL and sets the second array with RGB equivalents as determined by hsl_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("hsl_to_rgb", &ip::hsl_to_rgb<uint16_t>, (arg("hsl"), arg("rgb")), "Takes a 3-dimensional array encoded as HSL and sets the second array with RGB equivalents as determined by hsl_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("hsl_to_rgb", &ip::hsl_to_rgb<float>, (arg("hsl"), arg("rgb")), "Takes a 3-dimensional array encoded as HSL and sets the second array with RGB equivalents as determined by hsl_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("rgb_to_yuv", &ip::rgb_to_yuv<uint8_t>, (arg("rgb"), arg("yuv")), "Takes a 3-dimensional array encoded as RGB and sets the second array with YUV (Y'CbCr) equivalents as determined by rgb_to_yuv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("rgb_to_yuv", &ip::rgb_to_yuv<uint16_t>, (arg("rgb"), arg("yuv")), "Takes a 3-dimensional array encoded as RGB and sets the second array with YUV (Y'CbCr) equivalents as determined by rgb_to_yuv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("rgb_to_yuv", &ip::rgb_to_yuv<float>, (arg("rgb"), arg("yuv")), "Takes a 3-dimensional array encoded as RGB and sets the second array with YUV (Y'CbCr) equivalents as determined by rgb_to_yuv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("yuv_to_rgb", &ip::yuv_to_rgb<uint8_t>, (arg("yuv"), arg("rgb")), "Takes a 3-dimensional array encoded as YUV (Y'CbCr) and sets the second array with RGB equivalents as determined by yuv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("yuv_to_rgb", &ip::yuv_to_rgb<uint16_t>, (arg("yuv"), arg("rgb")), "Takes a 3-dimensional array encoded as YUV (Y'CbCr) and sets the second array with RGB equivalents as determined by yuv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("yuv_to_rgb", &ip::yuv_to_rgb<float>, (arg("yuv"), arg("rgb")), "Takes a 3-dimensional array encoded as YUV (Y'CbCr) and sets the second array with RGB equivalents as determined by yuv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported.");
  def("rgb_to_gray", &ip::rgb_to_gray<uint8_t>, (arg("rgb"), arg("gray")), "Takes a 3-dimensional array encoded as RGB and sets the second array with gray equivalents as determined by rgb_to_gray_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array is a 2D array with the same element type. The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported");
  def("rgb_to_gray", &ip::rgb_to_gray<uint16_t>, (arg("rgb"), arg("gray")), "Takes a 3-dimensional array encoded as RGB and sets the second array with gray equivalents as determined by rgb_to_gray_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array is a 2D array with the same element type. The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported");
  def("rgb_to_gray", &ip::rgb_to_gray<float>, (arg("rgb"), arg("gray")), "Takes a 3-dimensional array encoded as RGB and sets the second array with gray equivalents as determined by rgb_to_gray_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array is a 2D array with the same element type. The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported");
  def("gray_to_rgb", &ip::gray_to_rgb<uint8_t>, (arg("gray"), arg("rgb")), "Takes a 2-dimensional array encoded as grays and sets the second array with RGB equivalents as determined by gray_to_rgb_one(). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported");
  def("gray_to_rgb", &ip::gray_to_rgb<uint16_t>, (arg("gray"), arg("rgb")), "Takes a 2-dimensional array encoded as grays and sets the second array with RGB equivalents as determined by gray_to_rgb_one(). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported");
  def("gray_to_rgb", &ip::gray_to_rgb<float>, (arg("gray"), arg("rgb")), "Takes a 2-dimensional array encoded as grays and sets the second array with RGB equivalents as determined by gray_to_rgb_one(). The output array will be automatically resized if required. WARNING: As of this time only C-style storage arrays are supported");
}
