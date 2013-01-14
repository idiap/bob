/**
 * @file ip/python/drawing.cc
 * @date Sun Jul 24 21:13:21 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds simple drawing primitives
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

#include "bob/core/python/ndarray.h"
#include "bob/ip/drawing.h"

using namespace boost::python;
namespace ip = bob::ip;
namespace tp = bob::python;
namespace ca = bob::core::array;

template <typename T>
static void inner_draw_point_ (tp::ndarray image, int x, int y, object color) {
  switch (image.type().nd) {
    case 2: 
      {
        blitz::Array<T,2> image_ = image.bz<T,2>();
        T tcolor = extract<T>(color);
        ip::draw_point_(image_, x, y, tcolor);
      }
      break;
    case 3:
      {
        blitz::Array<T,3> image_ = image.bz<T,3>();
        tuple c = extract<tuple>(color);
        T c0 = extract<T>(c[0]); 
        T c1 = extract<T>(c[1]); 
        T c2 = extract<T>(c[2]);
        boost::tuple<T,T,T> tcolor(c0, c1, c2);
        ip::draw_point_(image_, x, y, tcolor);
      }
      break;
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", image.type().str().c_str());
  }
}

static void draw_point_ (tp::ndarray image, int x, int y, object color) {
  const ca::typeinfo& info = image.type();
  switch(info.dtype) {
    case ca::t_uint8: return inner_draw_point_<uint8_t>(image, x, y, color);
    case ca::t_uint16: return inner_draw_point_<uint16_t>(image, x, y, color);
    case ca::t_float64: return inner_draw_point_<double>(image, x, y, color);
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void inner_draw_point (tp::ndarray image, int x, int y, object color) {
  switch (image.type().nd) {
    case 2: 
      {
        blitz::Array<T,2> image_ = image.bz<T,2>();
        T tcolor = extract<T>(color);
        ip::draw_point(image_, x, y, tcolor);
      }
      break;
    case 3:
      {
        blitz::Array<T,3> image_ = image.bz<T,3>();
        tuple c = extract<tuple>(color);
        T c0 = extract<T>(c[0]); 
        T c1 = extract<T>(c[1]); 
        T c2 = extract<T>(c[2]);
        boost::tuple<T,T,T> tcolor(c0, c1, c2);
        ip::draw_point(image_, x, y, tcolor);
      }
      break;
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", image.type().str().c_str());
  }
}

static void draw_point (tp::ndarray image, int x, int y, object color) {
  const ca::typeinfo& info = image.type();
  switch(info.dtype) {
    case ca::t_uint8: return inner_draw_point<uint8_t>(image, x, y, color);
    case ca::t_uint16: return inner_draw_point<uint16_t>(image, x, y, color);
    case ca::t_float64: return inner_draw_point<double>(image, x, y, color);
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void inner_try_draw_point (tp::ndarray image, int x, int y, object color) {
  switch (image.type().nd) {
    case 2: 
      {
        blitz::Array<T,2> image_ = image.bz<T,2>();
        T tcolor = extract<T>(color);
        ip::try_draw_point(image_, x, y, tcolor);
      }
      break;
    case 3:
      {
        blitz::Array<T,3> image_ = image.bz<T,3>();
        tuple c = extract<tuple>(color);
        T c0 = extract<T>(c[0]); 
        T c1 = extract<T>(c[1]); 
        T c2 = extract<T>(c[2]);
        boost::tuple<T,T,T> tcolor(c0, c1, c2);
        ip::try_draw_point(image_, x, y, tcolor);
      }
      break;
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", image.type().str().c_str());
  }
}

static void try_draw_point (tp::ndarray image, int x, int y, object color) {
  const ca::typeinfo& info = image.type();
  switch(info.dtype) {
    case ca::t_uint8: return inner_try_draw_point<uint8_t>(image, x, y, color);
    case ca::t_uint16: return inner_try_draw_point<uint16_t>(image, x, y, color);
    case ca::t_float64: return inner_try_draw_point<double>(image, x, y, color);
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void inner_draw_line (tp::ndarray image, int x1, int y1, int x2, int y2, object color) {
  switch (image.type().nd) {
    case 2: 
      {
        blitz::Array<T,2> image_ = image.bz<T,2>();
        T tcolor = extract<T>(color);
        ip::draw_line(image_, x1, y1, x2, y2, tcolor);
      }
      break;
    case 3:
      {
        blitz::Array<T,3> image_ = image.bz<T,3>();
        tuple c = extract<tuple>(color);
        T c0 = extract<T>(c[0]); 
        T c1 = extract<T>(c[1]); 
        T c2 = extract<T>(c[2]);
        boost::tuple<T,T,T> tcolor(c0, c1, c2);
        ip::draw_line(image_, x1, y1, x2, y2, tcolor);
      }
      break;
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", image.type().str().c_str());
  }
}

static void draw_line (tp::ndarray image, int x1, int y1, int x2, int y2, object color) {
  const ca::typeinfo& info = image.type();
  switch(info.dtype) {
    case ca::t_uint8: return inner_draw_line<uint8_t>(image, x1, y1, x2, y2, color);
    case ca::t_uint16: return inner_draw_line<uint16_t>(image, x1, y1, x2, y2, color);
    case ca::t_float64: return inner_draw_line<double>(image, x1, y1, x2, y2, color);
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void inner_draw_cross (tp::ndarray image, int x, int y, int radius, object color) {
  switch (image.type().nd) {
    case 2: 
      {
        blitz::Array<T,2> image_ = image.bz<T,2>();
        T tcolor = extract<T>(color);
        ip::draw_cross(image_, x, y, radius, tcolor);
      }
      break;
    case 3:
      {
        blitz::Array<T,3> image_ = image.bz<T,3>();
        tuple c = extract<tuple>(color);
        T c0 = extract<T>(c[0]); 
        T c1 = extract<T>(c[1]); 
        T c2 = extract<T>(c[2]);
        boost::tuple<T,T,T> tcolor(c0, c1, c2);
        ip::draw_cross(image_, x, y, radius, tcolor);
      }
      break;
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", image.type().str().c_str());
  }
}

static void draw_cross (tp::ndarray image, int x, int y, int radius, 
    object color) {
  const ca::typeinfo& info = image.type();
  switch(info.dtype) {
    case ca::t_uint8: return inner_draw_cross<uint8_t>(image, x, y, radius, color);
    case ca::t_uint16: return inner_draw_cross<uint16_t>(image, x, y, radius, color);
    case ca::t_float64: return inner_draw_cross<double>(image, x, y, radius, color);
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void inner_draw_cross_plus (tp::ndarray image, int x, int y, int radius, object color) {
  switch (image.type().nd) {
    case 2: 
      {
        blitz::Array<T,2> image_ = image.bz<T,2>();
        T tcolor = extract<T>(color);
        ip::draw_cross_plus(image_, x, y, radius, tcolor);
      }
      break;
    case 3:
      {
        blitz::Array<T,3> image_ = image.bz<T,3>();
        tuple c = extract<tuple>(color);
        T c0 = extract<T>(c[0]); 
        T c1 = extract<T>(c[1]); 
        T c2 = extract<T>(c[2]);
        boost::tuple<T,T,T> tcolor(c0, c1, c2);
        ip::draw_cross_plus(image_, x, y, radius, tcolor);
      }
      break;
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", image.type().str().c_str());
  }
}

static void draw_cross_plus (tp::ndarray image, int x, int y, int radius, 
    object color) {
  const ca::typeinfo& info = image.type();
  switch(info.dtype) {
    case ca::t_uint8: return inner_draw_cross_plus<uint8_t>(image, x, y, radius, color);
    case ca::t_uint16: return inner_draw_cross_plus<uint16_t>(image, x, y, radius, color);
    case ca::t_float64: return inner_draw_cross_plus<double>(image, x, y, radius, color);
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void inner_draw_box (tp::ndarray image, int x, int y, int w, int h, object color) {
  switch (image.type().nd) {
    case 2: 
      {
        blitz::Array<T,2> image_ = image.bz<T,2>();
        T tcolor = extract<T>(color);
        ip::draw_box(image_, x, y, w, h, tcolor);
      }
      break;
    case 3:
      {
        blitz::Array<T,3> image_ = image.bz<T,3>();
        tuple c = extract<tuple>(color);
        T c0 = extract<T>(c[0]); 
        T c1 = extract<T>(c[1]); 
        T c2 = extract<T>(c[2]);
        boost::tuple<T,T,T> tcolor(c0, c1, c2);
        ip::draw_box(image_, x, y, w, h, tcolor);
      }
      break;
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", image.type().str().c_str());
  }
}

static void draw_box (tp::ndarray image, int x, int y, int w, int h, 
    object color) {
  const ca::typeinfo& info = image.type();
  switch(info.dtype) {
    case ca::t_uint8: return inner_draw_box<uint8_t>(image, x, y, w, h, color);
    case ca::t_uint16: return inner_draw_box<uint16_t>(image, x, y, w, h, color);
    case ca::t_float64: return inner_draw_box<double>(image, x, y, w, h, color);
    default: PYTHON_ERROR(TypeError, "drawing operation does not support '%s'", info.str().c_str());
  }
}

void bind_ip_drawing() {
  def("draw_point_", &draw_point_, (arg("image"), arg("x"), arg("y"), arg("color")), "Draws a point in an image. No checks, if you try to access an area outside the image using this method, you may trigger a segmentation fault. This method supports both grayscale (2D) or color RGB (3D) images. Depending on your image type, select an appropriate color value: a single gray value for 2D grayscale images or a 3-tuple containing the RGB color to set during drawing.");
  def("draw_point", &draw_point, (arg("image"), arg("x"), arg("y"), arg("color")), "Draws a point in the given image. Trying to access outside the image range will trigger an IndexError. This method supports both grayscale (2D) or color RGB (3D) images. Depending on your image type, select an appropriate color value: a single gray value for 2D grayscale images or a 3-tuple containing the RGB color to set during drawing.");
  def("try_draw_point", try_draw_point, (arg("image"), arg("x"), arg("y"), arg("color")), "Tries to draw a point at the given image. If the point is out of range, just ignores the request. This is what is used for drawing lines. This method supports both grayscale (2D) or color RGB (3D) images. Depending on your image type, select an appropriate color value: a single gray value for 2D grayscale images or a 3-tuple containing the RGB color to set during drawing.");
  def("draw_line", &draw_line, (arg("image"), arg("x1"), arg("y1"), arg("x2"), arg("y2"), arg("color")), "Draws a line between two points p1(x1,y1) and p2(x2,y2).  This function is based on the Bresenham's line algorithm and is highly optimized to be able to draw lines very quickly. There is no floating point arithmetic nor multiplications and divisions involved. Only addition, subtraction and bit shifting are used.\n\nThe line may go out of the image bounds in which case such points (lying outside the image boundary are ignored).\n\nReferences: http://en.wikipedia.org/wiki/Bresenham's_line_algorithm. This method supports both grayscale (2D) or color RGB (3D) images. Depending on your image type, select an appropriate color value: a single gray value for 2D grayscale images or a 3-tuple containing the RGB color to set during drawing.");
  def("draw_cross", &draw_cross, (arg("image"), arg("x"), arg("y"), arg("radius"), arg("color")), "Draws a cross with a given radius and color at the image. Uses the draw_line() primitive above. The cross will look like an 'x' and not like a '+'. To get a '+' sign, use the draw_cross_plus() variant. This method supports both grayscale (2D) or color RGB (3D) images. Depending on your image type, select an appropriate color value: a single gray value for 2D grayscale images or a 3-tuple containing the RGB color to set during drawing.");
  def("draw_cross_plus", &draw_cross_plus, (arg("image"), arg("x"), arg("y"), arg("radius"), arg("color")), "Draws a cross with a given radius and color at the image. Uses the draw_line() primitive above. The cross will look like an '+' and not like a 'x'. To get a 'x' sign, use the draw_cross() variant. This method supports both grayscale (2D) or color RGB (3D) images. Depending on your image type, select an appropriate color value: a single gray value for 2D grayscale images or a 3-tuple containing the RGB color to set during drawing.");
  def("draw_box", &draw_box, (arg("image"), arg("x"), arg("y"), arg("width"), arg("height"), arg("color")), "Draws a box at the image using the draw_line() primitive. This method supports both grayscale (2D) or color RGB (3D) images. Depending on your image type, select an appropriate color value: a single gray value for 2D grayscale images or a 3-tuple containing the RGB color to set during drawing.");
}
