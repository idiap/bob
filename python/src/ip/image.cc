/**
 * @file src/ip/image.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Video framework to python
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/Object.h"
#include "core/Tensor.h"
#include "ip/vision.h"
#include "ip/Color.h"
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

static bool save_image(const Torch::Image& i, const char* filename)
{
  Torch::xtprobeImageFile loader;
  return loader.save(i, filename);
}

void bind_ip_image()
{
  class_<Torch::Image, boost::shared_ptr<Torch::Image>, bases<Torch::Object, Torch::ShortTensor> >("Image", init<optional<int, int, int> >((arg("width"), arg("height"), arg("planes"))))
    .def("__init__", make_constructor(load_image))
    .def("save", &save_image, (arg("self"), arg("filename")), "Saves the image using a standard format, guessed by the filename (e.g. jpg, ppm, pgm, tif or gif)")
    .def("getWidth", &Torch::Image::getWidth, arg("self"), "Returns the image width")
    .def("getHeight", &Torch::Image::getHeight, arg("self"), "Returns the image height")
    .def("getNPlanes", &Torch::Image::getNPlanes, arg("self"), "Returns the number of planes currently in the image")
    .def("resize", &Torch::Image::resize, (arg("self"), arg("width"), arg("height"), arg("planes")), "Resizes the current image")
    .def("copyFromImage", (bool (Torch::Image::*)(const Torch::Image&))&Torch::Image::copyFrom, (arg("self"), arg("image")), "Copy the data from the other image converting it appropriately taking into consideration the number of planes")
    .def("copyFromTensor", (bool (Torch::Image::*)(const Torch::Tensor&))&Torch::Image::copyFrom, (arg("self"), arg("tensor")), "Copy data from a tensor (with the exact same dimensions)")
    .def("drawPixel", &Torch::Image::drawPixel)
    .def("drawLine", &Torch::Image::drawLine)
    .def("drawCross", &Torch::Image::drawLine)
    .def("drawRect", (void (Torch::Image::*)(int, int, int, int, const Torch::Color&))&Torch::Image::drawRect)
    .def("drawRect", (void (Torch::Image::*)(const Torch::sRect2D&, const Torch::Color&))&Torch::Image::drawRect)
    .def("_toGray", &image_make_gray, arg("self"), "Creates a copy of this image in grayscale\n\nIf this image is already in grayscale, just copies the data.")
    .def("_toRGB", &image_make_rgb, arg("self"), "Creates a copy of this image in RGB\n\nIf this image is already in colors (i.e. has 3 planes), just copies the data.")
    ;
}

