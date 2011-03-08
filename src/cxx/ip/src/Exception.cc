/**
 * @file src/cxx/ip/src/Exception.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Implementation of the exceptions used throughout the IP subsystem of
 * Torch.
 */

#include <boost/format.hpp>
#include "ip/Exception.h"

namespace ip = Torch::ip;
namespace core = Torch::core;

ip::Exception::Exception() throw() {
}

ip::Exception::~Exception() throw() {
}

const char* ip::Exception::what() const throw() {
  static const char* what_string = "Generic ip::Exception: user \
    specialization has not overwritten what() or is throwing an object of \
    this class (in which case, please fix it!)";
  return what_string;
}

ip::ParamOutOfBoundaryError::ParamOutOfBoundaryError(
  const std::string& paramname, const bool larger, const int value, 
  const int limit) throw(): 
    m_paramname(paramname), m_larger(larger), m_value(value), m_limit(limit) 
{
}

ip::ParamOutOfBoundaryError::~ParamOutOfBoundaryError() throw() {
}

const char* ip::ParamOutOfBoundaryError::what() const throw() {
  try {
    static const char* s_larger = "larger";
    static const char* s_smaller = "smaller";
    const char* s_selected = ( m_larger ? s_larger : s_smaller);
    boost::format message(
      "Parameter '%s' (value=%d) is %s than the limit %d.");
    message % m_paramname;
    message % m_value;
    message % s_selected;
    message % m_limit;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "ip::ParamOutOfBoundaryError: cannot \
      format, exception raised";
    return emergency;
  }
}

