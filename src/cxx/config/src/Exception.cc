/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun  6 Mar 19:50:39 2011 
 *
 * @brief Implementation of Configuration exceptions 
 */

#include <boost/format.hpp>
#include "config/Exception.h"

namespace conf = Torch::config;
namespace bp = boost::python;

conf::Exception::Exception() throw() {
}

conf::Exception::~Exception() throw() {
}

const char* conf::Exception::what() const throw() {
 static const char* what_string = "Generic config::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

conf::KeyError::KeyError(const std::string& key) throw(): m_key(key) {
}

conf::KeyError::~KeyError() throw() {
}

const char* conf::KeyError::what() const throw() {
  try {
    boost::format message("Unknown Configuration key '%s'");
    message % m_key;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "config::KeyError: cannot format, exception raised";
    return emergency;
  }
}

conf::UnsupportedConversion::UnsupportedConversion
(const std::string& varname, const std::type_info& cxx_type,
 boost::python::object o) throw():
  m_varname(varname),
  m_typeinfo(cxx_type),
  m_object(o) 
{
}

conf::UnsupportedConversion::~UnsupportedConversion() throw() {
}

const char* conf::UnsupportedConversion::what() const throw() {
  try {
    boost::format message("The Torch configuration parameter pointed by key '%s' cannot be extract into an object of C++ runtime-type '%s' because it is of type '%s' and that conversion is not supported. If you think this is an error, please submit a ticket and include this message.");
    message % m_varname;
    message % m_typeinfo.name();
    std::string s = bp::extract<std::string>(m_object.attr("__class__").attr("__name__"));
    message % s;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "config::UnsupportedConversion: cannot format, exception raised";
    return emergency;
  }
}

conf::PythonError::PythonError() throw() {
}

conf::PythonError::~PythonError() throw() {
}

const char* conf::PythonError::what() const throw() {
  static const char* emergency = "config::PythonError";
  return emergency;
}

