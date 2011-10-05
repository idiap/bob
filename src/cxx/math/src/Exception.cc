/**
  * @file src/cxx/math/src/Exception.cc
  * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
  * @date Mon 06 June 11:06:07 2011
  *
  * @brief Implements a generic exception for the math subsystem of Torch 
 *
 * @brief  
 */

#include "math/Exception.h"
#include <boost/format.hpp>

namespace math = Torch::math;

math::Exception::Exception() throw() {
}

math::Exception::~Exception() throw() {
}

const char* math::Exception::what() const throw() {
 static const char* what_string = "Generic math::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

math::LapackError::LapackError(const std::string& msg) throw(): m_submessage(msg) {
}

math::LapackError::~LapackError() throw() {
}

const char* math::LapackError::what() const throw() {
  try {
    boost::format message("Error when executing a LAPACK function. '%s'");
    message % m_submessage;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "math::LapackError: cannot format, exception raised";
    return emergency;
  }
}

math::NorminvPNotInRangeError::NorminvPNotInRangeError(const double p) throw(): m_p(p) {
}

math::NorminvPNotInRangeError::~NorminvPNotInRangeError() throw() {
}

const char* math::NorminvPNotInRangeError::what() const throw() {
  try {
    boost::format message("The parameter p of the norminv function has value '%f', not in the range [0,1].");
    message % m_p;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "math::NorminvPNotInRangeError: cannot format, exception raised";
    return emergency;
  }
}

