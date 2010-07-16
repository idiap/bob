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

void bind_ip_image()
{
  class_<Torch::Image, boost::shared_ptr<Torch::Image>, bases<Torch::Object, Torch::ShortTensor> >("Image", no_init)
    .def(init<optional<int, int, int> >())
    .def("getWidth", &Torch::Image::getWidth)
    .def("getHeight", &Torch::Image::getHeight)
    .def("getNPlanes", &Torch::Image::getNPlanes)
    .def("resize", &Torch::Image::resize)
    .def("copyFromImage", (bool (Torch::Image::*)(const Torch::Image&))&Torch::Image::copyFrom)
    .def("copyFromTensor", (bool (Torch::Image::*)(const Torch::Tensor&))&Torch::Image::copyFrom)
    .def("drawPixel", &Torch::Image::drawPixel)
    .def("drawLine", &Torch::Image::drawLine)
    .def("drawCross", &Torch::Image::drawLine)
    .def("drawRect", (void (Torch::Image::*)(int, int, int, int, const Torch::Color&))&Torch::Image::drawRect)
    .def("drawRect", (void (Torch::Image::*)(const Torch::sRect2D&, const Torch::Color&))&Torch::Image::drawRect)
    .def("_toGray", &image_make_gray)
    .def("_toRGB", &image_make_rgb)
    ;
}

