/**
  * @file src/cxx/machine/src/Exception.cc
  * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
  * @date Fri 13 May 19:27:09 2011
  *
  * @brief Implements a generic exception for the machine subsystem of Torch 
 *
 * @brief  
 */

#include "machine/Exception.h"
#include <boost/format.hpp>

namespace machine = Torch::machine;

machine::Exception::Exception() throw() {
}

machine::Exception::~Exception() throw() {
}

const char* machine::Exception::what() const throw() {
 static const char* what_string = "Generic machine::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

machine::NInputsMismatch::NInputsMismatch(const int n1, const int n2) throw(): m_n_inputs1(n1), m_n_inputs2(n2) {
}

machine::NInputsMismatch::~NInputsMismatch() throw() {
}

const char* machine::NInputsMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the number of inputs: '%d' vs '%d'.");
    message % m_n_inputs1;
    message % m_n_inputs2;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "machine::NInputsMismatch: cannot format, exception raised";
    return emergency;
  }
}

machine::NOutputsMismatch::NOutputsMismatch(const int n1, const int n2) throw(): m_n_outputs1(n1), m_n_outputs2(n2) {
}

machine::NOutputsMismatch::~NOutputsMismatch() throw() {
}

const char* machine::NOutputsMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the number of outputs: '%d' vs '%d'.");
    message % m_n_outputs1;
    message % m_n_outputs2;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "machine::NOutputsMismatch: cannot format, exception raised";
    return emergency;
  }
}
