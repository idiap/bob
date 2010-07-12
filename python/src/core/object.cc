/**
 * @file src/core/object.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Core object type into python 
 */

#include <boost/python.hpp>

#include "core/Object.h"

using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addBOption_overloads, addBOption, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addIOption_overloads, addIOption, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addFOption_overloads, addFOption, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addDOption_overloads, addDOption, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addSOption_overloads, addSOption, 2, 3);

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getBOption_overloads, getBOption, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getIOption_overloads, getIOption, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getFOption_overloads, getFOption, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getDOption_overloads, getDOption, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getSOption_overloads, getSOption, 1, 2);

void bind_core_object()
{
  class_<Torch::Object>("Object", init<>())
    .def("addBOption", &Torch::Object::addBOption, addBOption_overloads())
    .def("addIOption", &Torch::Object::addIOption, addIOption_overloads())
    .def("addFOption", &Torch::Object::addFOption, addFOption_overloads())
    .def("addDOption", &Torch::Object::addDOption, addDOption_overloads())
    .def("addSOption", &Torch::Object::addSOption, addSOption_overloads())
    
    .def("setBOoption", &Torch::Object::addBOption)
    .def("setIOoption", &Torch::Object::addIOption)
    .def("setFOoption", &Torch::Object::addFOption)
    .def("setDOoption", &Torch::Object::addDOption)
    .def("setSOoption", &Torch::Object::addSOption)
    
    .def("getBOption", &Torch::Object::getBOption, getBOption_overloads())
    .def("getIOption", &Torch::Object::getIOption, getIOption_overloads())
    .def("getFOption", &Torch::Object::getFOption, getFOption_overloads())
    .def("getDOption", &Torch::Object::getDOption, getDOption_overloads())
    .def("getSOption", &Torch::Object::getSOption, getSOption_overloads())
    ;
}
