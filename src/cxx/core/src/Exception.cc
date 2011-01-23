/**
 * @file src/cxx/core/src/Exception.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the Exception class. Makes sure we have at least one
 * virtual method implemented in a cxx file so that the pythonic bindings work
 * as expected.
 */

#include "core/Exception.h"

Torch::core::Exception::Exception() throw() {
}

Torch::core::Exception::Exception(const Torch::core::Exception&) throw() {
}

Torch::core::Exception::~Exception() throw() {
}

const char* Torch::core::Exception::what() const throw() {
 static const char* what_string = "Torch::core::Exception";
 return what_string;
}
