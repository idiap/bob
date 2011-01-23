/**
 * @file src/python/core/src/object.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Core object type into python 
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

//a pythonic implementation of Torch::Object::getVariables()

static const Torch::Variable* get_variable(Torch::Object& o, size_t i) {
  const Torch::Variable* var = o.getVariables();
  size_t nvar = o.getNvariables();
  if (i < nvar) return &var[i];
  return 0;
}

static const char* get_name(Torch::Variable& v) {
  return v.m_name;
}

static const char* get_help(Torch::Variable& v) {
  return v.m_help;
}

void bind_core_object()
{
  enum_<Torch::Variable::Type>("VariableType") 
      .value("Nothing", Torch::Variable::TypeNothing)
      .value("Bool", Torch::Variable::TypeBool)
      .value("Int", Torch::Variable::TypeInt)
      .value("Float", Torch::Variable::TypeFloat)
      .value("Double", Torch::Variable::TypeDouble)
      .value("String", Torch::Variable::TypeString)
      .value("IntArray", Torch::Variable::TypeIntArray)
      .value("FloatArray", Torch::Variable::TypeFloatArray)
      .value("DoubleArray", Torch::Variable::TypeDoubleArray)
      ;

  class_<Torch::Variable>("Variable", "Options are synthetized as Variables, internally.", init<const Torch::Variable&>())
    .def_readonly("type", &Torch::Variable::m_type)
    .add_property("name", &get_name)
    .add_property("help", &get_help)
    .def_readonly("n", &Torch::Variable::m_n_values)
    ;

  class_<Torch::Object>("Object", "This is the base class of almost all Torch types. It allows setting and getting options.", init<>())
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
    .def("nVariables", &Torch::Object::getNvariables)
    .def("variable", &get_variable, return_internal_reference<>())
    ;
}
