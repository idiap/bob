/**
 * @file src/python/ip/color.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds Color to python 
 */

#include <boost/python.hpp>

#include "ip/Color.h"

using namespace boost::python;

void bind_ip_color()
{
  class_<Torch::Color>("Color", init<>())
    .def(init<unsigned char, unsigned char, unsigned char, optional<const char*> >())
    .def(init<const char*>())
    .def("setGray", &Torch::Color::setGray)
    .def("setRGB", &Torch::Color::setRGB)
    .def("setYUV", &Torch::Color::setYUV)
    .def_readwrite("data0", &Torch::Color::data0)
    .def_readwrite("data1", &Torch::Color::data1)
    .def_readwrite("data2", &Torch::Color::data2)
    .def_readwrite("coding", &Torch::Color::coding)
    ;

  scope().attr("black") = &Torch::black;
  scope().attr("white") = &Torch::white;
  scope().attr("green") = &Torch::green;
  scope().attr("lightgreen") = &Torch::lightgreen;
  scope().attr("red") = &Torch::red;
  scope().attr("lightred") = &Torch::lightred;
  scope().attr("blue") = &Torch::blue;
  scope().attr("lightblue") = &Torch::lightblue;
  scope().attr("yellow") = &Torch::yellow;
  scope().attr("lightyellow") = &Torch::lightyellow;
  scope().attr("cyan") = &Torch::cyan;
  scope().attr("lightcyan") = &Torch::lightcyan;
  scope().attr("seagreen") = &Torch::seagreen;
  scope().attr("pink") = &Torch::pink;
  scope().attr("orange") = &Torch::orange;
}
