/**
 * @file src/python/core/src/exception.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the core extension into Python. Please note that, for each
 * exception type you only need to declare once the translator. All other
 * modules will benefit from it automatically.
 */

#include <Python.h>
#include "core/Exception.h"
#include "core/array_exception.h"
#include "core/convert_exception.h"
#include "core/python/exception.h"

using namespace Torch::python;
using namespace boost::python;

/**
 * This method is only useful to test exception throwing in Python code.
 */
static void throw_exception(void) {
  throw Torch::core::Exception();
}

void bind_core_exception() {

  // avoid binding std::except as boost::python uses it...
  // register_exception_translator<std::exception>(PyExc_RuntimeError);
  
  register_exception_translator<std::logic_error>(PyExc_RuntimeError);
  register_exception_translator<std::domain_error>(PyExc_RuntimeError);
  register_exception_translator<std::invalid_argument>(PyExc_TypeError);
  register_exception_translator<std::length_error>(PyExc_RuntimeError);
  register_exception_translator<std::out_of_range>(PyExc_IndexError);
  register_exception_translator<std::runtime_error>(PyExc_RuntimeError);
  register_exception_translator<std::range_error>(PyExc_IndexError);
  register_exception_translator<std::overflow_error>(PyExc_OverflowError);
  register_exception_translator<std::underflow_error>(PyExc_ArithmeticError);

  register_exception_translator<Torch::core::Exception>(PyExc_RuntimeError);

  // note: only register exceptions to which you need specific behavior not
  // covered by catching RuntimeError
 
  // just for tests...
  boost::python::def("throw_exception", &throw_exception);
}
