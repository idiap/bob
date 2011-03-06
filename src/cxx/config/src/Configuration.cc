/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat  5 Mar 20:26:56 2011 
 *
 * @brief Implementation of the Configuration main class
 */

#include "config/Configuration.h"

namespace conf = Torch::config;
namespace bp = boost::python;

conf::Configuration::Configuration(const std::string& s):
  m_py(conf::Python::instance()),
  m_dict()
{
  try {
    bp::object module = bp::import("__main__");
    bp::object name_space = module.attr("__dict__");
    name_space["__file__"] = bp::str(s.c_str());
    bp::exec_file(s.c_str(), name_space, m_dict);
  }
  catch (bp::error_already_set) {
    /**
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback); //this will clear the error

    bp::object exception_type(bp::handle<>(ptype));
    bp::object exception_value(bp::handle<>(pvalue));
    bp::object traceback(bp::handle<>(ptraceback));

    str exception_typename = bp::extract<str>(exception_type.attr("__name__"));

    throw conf::ParserError(bp::extrac<std::string>(exception_typename), 
        bp::extract<std::string>(pvalue));
    **/
    PyErr_Print();
    throw conf::PythonError();
  }
}

conf::Configuration::Configuration(): 
  m_py(conf::Python::instance()),
  m_dict()
{
}

conf::Configuration::Configuration(const conf::Configuration& other):
  m_py(conf::Python::instance()),
  m_dict(other.m_dict)
{
}

conf::Configuration::~Configuration ()
{
}

conf::Configuration& conf::Configuration::operator= (const conf::Configuration& other) {
  m_dict = bp::dict();
  m_dict.update(other.m_dict);
  return *this;
}

conf::Configuration& conf::Configuration::update (const conf::Configuration& other) {
  m_dict.update(other.m_dict);
  return *this;
}
