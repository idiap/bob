/**
 * @file src/cxx/core/src/reshape_exception.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Implements the Exceptions related to the reshape function.
 */

#include <boost/format.hpp>
#include "core/reshape_exception.h"

Torch::core::ReshapeDifferentNumberOfElements::ReshapeDifferentNumberOfElements( 
  const int expected, const int got) throw(): 
    m_expected(expected), m_got(got) 
{
}

Torch::core::ReshapeDifferentNumberOfElements::~ReshapeDifferentNumberOfElements() throw() {
}

const char* Torch::core::ReshapeDifferentNumberOfElements::what() const throw() {
  try {
    boost::format message(
      "The 2D dst array has '%d' elements whereas tje 2D src array as '%d' elements.");
    message % m_got;
    message % m_expected;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::DifferentNumberOfElements: cannot \
      format, exception raised";
    return emergency;
  }
}

