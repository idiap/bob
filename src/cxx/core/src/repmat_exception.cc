/**
 * @file src/cxx/core/src/repmat_exception.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Implements the Exceptions related to the repmat function.
 */

#include <boost/format.hpp>
#include "core/repmat_exception.h"

Torch::core::RepmatNonMultipleLength::RepmatNonMultipleLength( 
  const int src_dim, const int dst_dim) throw(): 
    m_src_dim(src_dim), m_dst_dim(dst_dim) 
{
}

Torch::core::RepmatNonMultipleLength::~RepmatNonMultipleLength() throw() {
}

const char* Torch::core::RepmatNonMultipleLength::what() const throw() {
  try {
    boost::format message(
      "The 2D src array has a dimension of length '%d', whereas the 2D dst\
      array has a corresponding dimension of length '%d', which is not a\
      multiple of '%d'.");
    message % m_src_dim;
    message % m_dst_dim;
    message % m_src_dim;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::RepmatNonMultipleLength: cannot \
      format, exception raised";
    return emergency;
  }
}

