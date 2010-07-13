/**
 * @file src/ip/image.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Video framework to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "core/Tensor.h"
#include "ip/vision.h"
#include "ip/Color.h"
#include "ip/Image.h"

using namespace boost::python;

void bind_ip_image()
{
  class_<Torch::Image, bases<Torch::Object, Torch::ShortTensor> >("Image", no_init)
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
    ;
}

