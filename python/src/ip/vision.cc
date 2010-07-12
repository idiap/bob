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
  class_<Torch::sPixRGB>("sPixRGB")
    .def_readwrite("r", &Torch::sPixRGB::r)
    .def_readwrite("g", &Torch::sPixRGB::g)
    .def_readwrite("b", &Torch::sPixRGB::b)
    ;

  class_<Torch::sPixYUV>("sPixYUV")
    .def_readwrite("y", &Torch::sPixYUV::y)
    .def_readwrite("u", &Torch::sPixYUV::u)
    .def_readwrite("v", &Torch::sPixYUV::v)
    ;

  class_<Torch::sPoint2D>("sPoint2D")
    .def_readwrite("x", &Torch::sPoint2D::x)
    .def_readwrite("y", &Torch::sPoint2D::y)
    ;

  class_<Torch::sPoint2Dpolar>("sPoint2Dpolar")
    .def_readwrite("rho", &Torch::sPoint2Dpolar::rho)
    .def_readwrite("theta", &Torch::sPoint2Dpolar::theta)
    ;

  class_<Torch::sRect2D>("sRect2D")
    .def_readwrite("x", &Torch::sRect2D::x)
    .def_readwrite("y", &Torch::sRect2D::y)
    .def_readwrite("w", &Torch::sRect2D::w)
    .def_readwrite("h", &Torch::sRect2D::h)
    ;

  class_<Torch::sSize>("sSize", init<optional<int, int> >())
    .def_readwrite("w", &Torch::sSize::w)
    .def_readwrite("h", &Torch::sSize::h)
    ;

  class_<Torch::sRect2Dpolar>("sRect2Dpolar")
    .def_readwrite("tl", &Torch::sRect2Dpolar::tl)
    .def_readwrite("tr", &Torch::sRect2Dpolar::tr)
    .def_readwrite("bl", &Torch::sRect2Dpolar::bl)
    .def_readwrite("br", &Torch::sRect2Dpolar::br)
    ;

  //class_<Torch::sPoly2D>("sPoly2D"); /* still not implemented */

  class_<Torch::sOcton>("sOcton")
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

  class_<Torch::sComplex>("sComplex")
    .def_readwrite("r", &Torch::sComplex::r)
    .def_readwrite("i", &Torch::sComplex::i)
    ;
}
