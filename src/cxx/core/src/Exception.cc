/**
 * @file src/cxx/core/src/Exception.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the Exception class. Makes sure we have at least one
 * virtual method implemented in a cxx file so that the pythonic bindings work
 * as expected.
 */

#include <boost/format.hpp>
#include "core/Exception.h"

Torch::core::Exception::Exception() throw() {
}

Torch::core::Exception::Exception(const Torch::core::Exception&) throw() {
}

Torch::core::Exception::~Exception() throw() {
}

const char* Torch::core::Exception::what() const throw() {
 static const char* what_string = "Generic core::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

Torch::core::DeprecationError::DeprecationError(const std::string& op) throw(): 
  m_op(op) {
}

Torch::core::DeprecationError::~DeprecationError() throw() {
}

const char* Torch::core::DeprecationError::what() const throw() {
  try {
    boost::format message("Deprecated operation: %s");
    message % m_op;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::DeprecationError: cannot format, exception raised";
    return emergency;
  }
}

