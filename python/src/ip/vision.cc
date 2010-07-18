/**
 * @file src/python/ip/vision.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Geometric constructions from vision.h 
 */

#include <boost/python.hpp>

#include "ip/vision.h"

using namespace boost::python;

void bind_ip_vision()
{
  class_<Torch::sPixRGB>("sPixRGB", "An RGB pixel representation as a 3-element vector")
    .def_readwrite("r", &Torch::sPixRGB::r)
    .def_readwrite("g", &Torch::sPixRGB::g)
    .def_readwrite("b", &Torch::sPixRGB::b)
    ;

  class_<Torch::sPixYUV>("sPixYUV", "An YUV pixel representation as a 3-element vector")
    .def_readwrite("y", &Torch::sPixYUV::y)
    .def_readwrite("u", &Torch::sPixYUV::u)
    .def_readwrite("v", &Torch::sPixYUV::v)
    ;

  class_<Torch::sPoint2D>("sPoint2D", "A point in a 2-dimensional plane as a 2-element vector")
    .def_readwrite("x", &Torch::sPoint2D::x)
    .def_readwrite("y", &Torch::sPoint2D::y)
    ;

  class_<Torch::sPoint2Dpolar>("sPoint2Dpolar", "A point in a 2-dimensional plane in polar coordinates")
    .def_readwrite("rho", &Torch::sPoint2Dpolar::rho)
    .def_readwrite("theta", &Torch::sPoint2Dpolar::theta)
    ;

  class_<Torch::sRect2D>("sRect2D", "A rectangle in a 2-dimensional plane using cartesian coordinates")
    .def_readwrite("x", &Torch::sRect2D::x)
    .def_readwrite("y", &Torch::sRect2D::y)
    .def_readwrite("w", &Torch::sRect2D::w)
    .def_readwrite("h", &Torch::sRect2D::h)
    ;

  class_<Torch::sSize>("sSize", "A window size", init<optional<int, int> >((arg("width"), arg("height")), "Constructs a new window out of a width and a height value.")) 
    .def_readwrite("w", &Torch::sSize::w)
    .def_readwrite("h", &Torch::sSize::h)
    ;

  class_<Torch::sRect2Dpolar>("sRect2Dpolar", "A rectangle in a 2-dimensional plane using polar coordinates")
    .def_readwrite("tl", &Torch::sRect2Dpolar::tl)
    .def_readwrite("tr", &Torch::sRect2Dpolar::tr)
    .def_readwrite("bl", &Torch::sRect2Dpolar::bl)
    .def_readwrite("br", &Torch::sRect2Dpolar::br)
    ;

  //class_<Torch::sPoly2D>("sPoly2D"); /* still not implemented */

  class_<Torch::sOcton>("sOcton", "An octagon")
    .def_readwrite("surface", &Torch::sOcton::surface)
    .def_readwrite("height", &Torch::sOcton::height)
    .def_readwrite("width", &Torch::sOcton::width)
    .def_readwrite("cg", &Torch::sOcton::cg)
    .def_readwrite("y_min", &Torch::sOcton::y_min)
    .def_readwrite("y_max", &Torch::sOcton::y_max)
    .def_readwrite("x_min", &Torch::sOcton::x_min)
    .def_readwrite("x_max", &Torch::sOcton::x_max)
    .def_readwrite("ypx_min", &Torch::sOcton::ypx_min)
    .def_readwrite("ypx_max", &Torch::sOcton::ypx_max)
    .def_readwrite("ymx_min", &Torch::sOcton::ymx_min)
    .def_readwrite("ymx_max", &Torch::sOcton::ymx_max)
    ;

  class_<Torch::sComplex>("sComplex", "A complex number representation")
    .def_readwrite("r", &Torch::sComplex::r)
    .def_readwrite("i", &Torch::sComplex::i)
    ;
}
