/**
 * @file python/old/ip/src/image.cc
 * @date Sat Apr 30 18:41:25 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Video framework to python
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

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/Exception.h"
#include "core/Object.h"
#include "core/Tensor.h"
#include "ip/vision.h"
#include "ip/OldColor.h"
#include "ip/Image.h"
#include "ip/ImageFile.h"
#include "ip/xtprobeImageFile.h"

using namespace boost::python;

/**
 * Converts an image from any format into grayscale.
 */
static boost::shared_ptr<Torch::Image> image_make_gray(const Torch::Image& i)
{
  boost::shared_ptr<Torch::Image> retval(new Torch::Image(i.getWidth(),
        i.getHeight(), 1));
  retval->copyFrom(i);
  return retval;
}

static boost::shared_ptr<Torch::Image> image_make_rgb(const Torch::Image& i)
{
  boost::shared_ptr<Torch::Image> retval(new Torch::Image(i.getWidth(),
        i.getHeight(), 3));
  retval->copyFrom(i);
  return retval;
}

static boost::shared_ptr<Torch::Image> load_image(const char* filename)
{
  Torch::Image* retval = new Torch::Image;
  Torch::xtprobeImageFile loader;
  if (loader.load(*retval, filename))
    return boost::shared_ptr<Torch::Image>(retval);
  return boost::shared_ptr<Torch::Image>();
}

static boost::shared_ptr<Torch::Image> from_tensor(const Torch::Tensor& t)
{
  Torch::Image* retval = new Torch::Image(t.size(1), t.size(0), t.size(2));
  if (retval->copyFrom(t)) return boost::shared_ptr<Torch::Image>(retval);
  return boost::shared_ptr<Torch::Image>();
}

static bool save_image(const Torch::Image& i, const char* filename)
{
  Torch::xtprobeImageFile loader;
  return loader.save(i, filename);
}

static bool reload_image(Torch::Image& i, const char* filename)
{
  Torch::xtprobeImageFile loader;
  return loader.load(i, filename);
}

static void inplace_abs(Torch::Image& self) {
  for (int i=0; i<self.getHeight(); ++i) {
    for (int j=0; j<self.getWidth(); ++j) {
      for (int k=0; k<self.getNPlanes(); ++k) {
        short& s = self(i, j, k);
        s = ((s<0)?-s:s);
      }
    }
  }
}

static void inplace_add(Torch::Image& self, const Torch::Image& other) {
  for (int i=0; i<self.getHeight(); ++i) {
    for (int j=0; j<self.getWidth(); ++j) {
      for (int k=0; k<self.getNPlanes(); ++k) {
        self(i, j, k) += other(i, j, k);
      }
    }
  }
}

static void inplace_subtract(Torch::Image& self, const Torch::Image& other) {
  for (int i=0; i<self.getHeight(); ++i) {
    for (int j=0; j<self.getWidth(); ++j) {
      for (int k=0; k<self.getNPlanes(); ++k) {
        self(i, j, k) -= other(i, j, k);
      }
    }
  }
}

static void inplace_reset(Torch::Image& self, short threshold, short value) {
  for (int i=0; i<self.getHeight(); ++i) {
    for (int j=0; j<self.getWidth(); ++j) {
      for (int k=0; k<self.getNPlanes(); ++k) {
        if (self(i, j, k) < threshold) self(i, j, k) = value;
      }
    }
  }
}

static double sum(Torch::Image& self) {
  double retval = 0;
  for (int i=0; i<self.getHeight(); ++i) {
    for (int j=0; j<self.getWidth(); ++j) {
      for (int k=0; k<self.getNPlanes(); ++k) {
        retval += self(i, j, k);
      }
    }
  }
  return retval;
}

/**
 * This method is only useful to test exception throwing in Python code.
 */
static void throw_exception(void) {
  throw Torch::core::Exception();
}

void bind_ip_image()
{
  class_<Torch::Image, boost::shared_ptr<Torch::Image>, bases<Torch::Object, Torch::ShortTensor> >("Image", init<optional<int, int, int> >((arg("width"), arg("height"), arg("planes"))))
    .def("__init__", make_constructor(load_image))
    .def("__init__", make_constructor(from_tensor))
    .def("save", &save_image, (arg("self"), arg("filename")), "Saves the image using a standard format, guessed by the filename (e.g. jpg, ppm, pgm, tif or gif)")
    .def("load", &reload_image, (arg("self"), arg("filename")), "Loads the image from a given file. Supports jpg, ppm, pgm, tif or gif.")
    .def("resize", &Torch::Image::resize, (arg("self"), arg("width"), arg("height"), arg("planes")), "Resizes the current image")
    .def("copyFromImage", (bool (Torch::Image::*)(const Torch::Image&))&Torch::Image::copyFrom, (arg("self"), arg("image")), "Copy the data from the other image converting it appropriately taking into consideration the number of planes")
    .def("copyFromTensor", (bool (Torch::Image::*)(const Torch::Tensor&))&Torch::Image::copyFrom, (arg("self"), arg("tensor")), "Copy data from a tensor (with the exact same dimensions)")
    .def("drawPixel", &Torch::Image::drawPixel, (arg("self"), arg("x"), arg("y"), arg("color")), "Draws a pixel in the image")
    .def("drawLine", &Torch::Image::drawLine, (arg("self"), arg("x1"), arg("y1"), arg("x2"), arg("y2"), arg("color")), "Draws a line from (x1,y1) till (x2,y2) in the image")
    .def("drawCross", &Torch::Image::drawLine, (arg("self"), arg("x"), arg("y"), arg("radius"), arg("color")), "Draws a cross with some radius centered in the given coordinates (x,y)")
    .def("drawRect", (void (Torch::Image::*)(int, int, int, int, const Torch::Color&))&Torch::Image::drawRect, (arg("self"), arg("x"), arg("y"), arg("width"), arg("height"), arg("color")), "Draws a (unfilled) rectangle into the image")
    .def("drawRect", (void (Torch::Image::*)(const Torch::sRect2D&, const Torch::Color&))&Torch::Image::drawRect, (arg("self"), arg("rectangle"), arg("color")), "Draws a (unfilled) rectangle into the image")
    .def("_toGray", &image_make_gray, arg("self"), "Creates a copy of this image in grayscale\n\nIf this image is already in grayscale, just copies the data.")
    .def("_toRGB", &image_make_rgb, arg("self"), "Creates a copy of this image in RGB\n\nIf this image is already in colors (i.e. has 3 planes), just copies the data.")
    .add_property("width", &Torch::Image::getWidth)
    .add_property("height", &Torch::Image::getHeight)
    .add_property("nplanes", &Torch::Image::getNPlanes)
    .def("__iadd__", &inplace_add, return_self<>(), (arg("self"), arg("other")), "Inplace addition of images")
    .def("__isub__", &inplace_subtract, return_self<>(), (arg("self"), arg("other")), "Inplace subtraction of images")
    .def("reset", &inplace_reset, return_self<>(), (arg("self"), arg("threshold"), arg("value")), "Sets values in the image that are smaller than 'threshold' to the given value")
    .def("abs", &inplace_abs, (arg("self")), "Applies pixel = abs(pixel) for each pixel in the image. Modifes the image inplace.")
    .def("sum", &sum, (arg("self")), "Returns the sum of all pixels in the image as a double value")
    .def("__call__", (short (Torch::Image::*)(long, long, long) const)&Torch::Image::get, (arg("self"), arg("y"), arg("x"), arg("plane")), "Returns a particular pixel from the image")
    .def("get", (short (Torch::Image::*)(long, long, long) const)&Torch::Image::get, (arg("self"), arg("y"), arg("x"), arg("plane")), "Returns a particular pixel from the image")
    ;
  def("throw_exception", &throw_exception);
}

