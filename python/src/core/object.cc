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
  class_<Torch::Object>("Object", init<>("This is the base class of almost all Torch types. It allows setting and getting options."))
    .def("addBOption", &Torch::Object::addBOption, addBOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named boolean option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    .def("addIOption", &Torch::Object::addIOption, addIOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named integer option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    .def("addFOption", &Torch::Object::addFOption, addFOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named float-point option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    .def("addDOption", &Torch::Object::addDOption, addDOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named double-precision option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    .def("addSOption", &Torch::Object::addSOption, addSOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named string option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    
    .def("setBOption", &Torch::Object::setBOption, (arg("self"), arg("name"), arg("value")), "Sets the named boolean option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    .def("setIOption", &Torch::Object::setIOption, (arg("self"), arg("name"), arg("value")), "Sets the named integer option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    .def("setFOption", &Torch::Object::setFOption, (arg("self"), arg("name"), arg("value")), "Sets the named float-point option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    .def("setDOption", &Torch::Object::setDOption, (arg("self"), arg("name"), arg("value")), "Sets the named double-precision option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    .def("setSOption", &Torch::Object::setSOption, (arg("self"), arg("name"), arg("value")), "Sets the named string option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    
    .def("getBOption", &Torch::Object::getBOption, getBOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named boolean values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    .def("getIOption", &Torch::Object::getIOption, getIOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named integer values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    .def("getFOption", &Torch::Object::getFOption, getFOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named float-point values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    .def("getDOption", &Torch::Object::getDOption, getDOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named double-precision values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    .def("getSOption", &Torch::Object::getSOption, getSOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named string values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    ;
}
