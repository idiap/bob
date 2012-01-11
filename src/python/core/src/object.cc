/**
 * @file python/core/src/object.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds the Core object type into python
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

#include "core/Object.h"

using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addBOption_overloads, addBOption, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addIOption_overloads, addIOption, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addFOption_overloads, addFOption, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addDOption_overloads, addDOption, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addSOption_overloads, addSOption, 2, 3)

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getBOption_overloads, getBOption, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getIOption_overloads, getIOption, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getFOption_overloads, getFOption, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getDOption_overloads, getDOption, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getSOption_overloads, getSOption, 1, 2)

//a pythonic implementation of bob::Object::getVariables()

static const bob::Variable* get_variable(bob::Object& o, size_t i) {
  const bob::Variable* var = o.getVariables();
  size_t nvar = o.getNvariables();
  if (i < nvar) return &var[i];
  return 0;
}

static const char* get_name(bob::Variable& v) {
  return v.m_name;
}

static const char* get_help(bob::Variable& v) {
  return v.m_help;
}

void bind_core_object()
{
  enum_<bob::Variable::Type>("VariableType") 
      .value("Nothing", bob::Variable::TypeNothing)
      .value("Bool", bob::Variable::TypeBool)
      .value("Int", bob::Variable::TypeInt)
      .value("Float", bob::Variable::TypeFloat)
      .value("Double", bob::Variable::TypeDouble)
      .value("String", bob::Variable::TypeString)
      .value("IntArray", bob::Variable::TypeIntArray)
      .value("FloatArray", bob::Variable::TypeFloatArray)
      .value("DoubleArray", bob::Variable::TypeDoubleArray)
      ;

  class_<bob::Variable>("Variable", "Options are synthetized as Variables, internally.", init<const bob::Variable&>())
    .def_readonly("type", &bob::Variable::m_type)
    .add_property("name", &get_name)
    .add_property("help", &get_help)
    .def_readonly("n", &bob::Variable::m_n_values)
    ;

  class_<bob::Object>("Object", "This is the base class of almost all bob types. It allows setting and getting options.", init<>())
    .def("addBOption", &bob::Object::addBOption, addBOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named boolean option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    .def("addIOption", &bob::Object::addIOption, addIOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named integer option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    .def("addFOption", &bob::Object::addFOption, addFOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named float-point option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    .def("addDOption", &bob::Object::addDOption, addDOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named double-precision option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    .def("addSOption", &bob::Object::addSOption, addSOption_overloads((arg("name"), arg("init_value"), arg("help")), "Adds the named string option to the object, with an initial value.\n\nThe exit status may be check to certify that the operation was successful."))
    
    .def("setBOption", &bob::Object::setBOption, (arg("self"), arg("name"), arg("value")), "Sets the named boolean option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    .def("setIOption", &bob::Object::setIOption, (arg("self"), arg("name"), arg("value")), "Sets the named integer option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    .def("setFOption", &bob::Object::setFOption, (arg("self"), arg("name"), arg("value")), "Sets the named float-point option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    .def("setDOption", &bob::Object::setDOption, (arg("self"), arg("name"), arg("value")), "Sets the named double-precision option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    .def("setSOption", &bob::Object::setSOption, (arg("self"), arg("name"), arg("value")), "Sets the named string option to the given value\n\nThe exit status may be checked to certify if the operation was successful.")
    
    .def("getBOption", &bob::Object::getBOption, getBOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named boolean values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    .def("getIOption", &bob::Object::getIOption, getIOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named integer values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    .def("getFOption", &bob::Object::getFOption, getFOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named float-point values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    .def("getDOption", &bob::Object::getDOption, getDOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named double-precision values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    .def("getSOption", &bob::Object::getSOption, getSOption_overloads((arg("name"), arg("ok")), "This method allows the retrieval of named string values.\n\nIf you pass a boolean object as second parameter, you can use it to check if the operation was successful."))
    .def("nVariables", &bob::Object::getNvariables)
    .def("variable", &get_variable, return_internal_reference<>())
    ;
}
