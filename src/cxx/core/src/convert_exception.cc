/**
 * @file src/cxx/core/src/convert_exception.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El-Shafey</a> 
 *
 * @brief Implements the Exception for the convert functions
 */

#include <boost/format.hpp>
#include "core/convert_exception.h"

Torch::core::ConvertZeroInputRange::ConvertZeroInputRange() throw() {
}

Torch::core::ConvertZeroInputRange::~ConvertZeroInputRange() throw() {
}

const char* Torch::core::ConvertZeroInputRange::what() const throw() {
  try {
    boost::format message("Cannot convert an array with a zero width input range.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::ConvertZeroInputRange: cannot format, exception raised";
    return emergency;
  }
}

Torch::core::ConvertInputAboveMaxRange::ConvertInputAboveMaxRange(const double v, const double m) throw():
  m_val(v), m_max(m)
{
}

Torch::core::ConvertInputAboveMaxRange::~ConvertInputAboveMaxRange() throw() {
}

const char* Torch::core::ConvertInputAboveMaxRange::what() const throw() {
  try {
    boost::format message("The value %f of the input array is above the maximum %f of the given input range.");
    message % m_val;
    message % m_max;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::ConvertInputAboveMaxRange: cannot format, exception raised";
    return emergency;
  }
}

Torch::core::ConvertInputBelowMinRange::ConvertInputBelowMinRange(const double v, const double m) throw():
  m_val(v), m_min(m)
{
}

Torch::core::ConvertInputBelowMinRange::~ConvertInputBelowMinRange() throw() {
}

const char* Torch::core::ConvertInputBelowMinRange::what() const throw() {
  try {
    boost::format message("The value %f of the input array is below the minimum %f of the given input range.");
    message % m_val;
    message % m_min;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::ConvertInputBelowMinRange: cannot format, exception raised";
    return emergency;
  }
}
