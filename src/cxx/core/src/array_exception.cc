/**
 * @file src/cxx/core/src/array_exception.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Implements the Exceptions related to the blitz++ multidimensional arrays
 */

#include <boost/format.hpp>
#include "core/array_exception.h"

Torch::core::NonZeroBaseError::NonZeroBaseError( const int dim, 
  const int base) throw(): 
    m_dim(dim), m_base(base) 
{
}

Torch::core::NonZeroBaseError::~NonZeroBaseError() throw() {
}

const char* Torch::core::NonZeroBaseError::what() const throw() {
  try {
    boost::format message(
      "The input array has dimension '%d' with a non-zero base index (base=%d).");
    message % m_dim;
    message % m_base;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::NonZeroBaseError: cannot \
      format, exception raised";
    return emergency;
  }
}


Torch::core::NonOneBaseError::NonOneBaseError( const int dim, 
  const int base) throw(): 
    m_dim(dim), m_base(base) 
{
}

Torch::core::NonOneBaseError::~NonOneBaseError() throw() {
}

const char* Torch::core::NonOneBaseError::what() const throw() {
  try {
    boost::format message(
      "The input array has dimension '%d' with a non-one base index (base=%d).");
    message % m_dim;
    message % m_base;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::NonOneBaseError: cannot \
      format, exception raised";
    return emergency;
  }
}


Torch::core::NonCContiguousError::NonCContiguousError() throw() {
}

Torch::core::NonCContiguousError::~NonCContiguousError() throw() {
}

const char* Torch::core::NonCContiguousError::what() const throw() {
  try {
    boost::format message(
      "The input array is not a C-style array (row-major order) stored in contiguous memory area.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::NonCContiguousError: cannot \
      format, exception raised";
    return emergency;
  }
}


Torch::core::NonFortranContiguousError::NonFortranContiguousError() throw() {
}

Torch::core::NonFortranContiguousError::~NonFortranContiguousError() throw() {
}

const char* Torch::core::NonFortranContiguousError::what() const throw() {
  try {
    boost::format message(
      "The input array is not a Fortran-style array (column-major order) stored in contiguous memory area.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::NonFortranContiguousError: cannot \
      format, exception raised";
    return emergency;
  }
}


Torch::core::UnexpectedShapeError::UnexpectedShapeError() throw() {
}

Torch::core::UnexpectedShapeError::~UnexpectedShapeError() throw() {
}

const char* Torch::core::UnexpectedShapeError::what() const throw() {
  try {
    boost::format message(
      "The array does not have the expected size.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::UnexpectedShapeError: cannot \
      format, exception raised";
    return emergency;
  }
}

