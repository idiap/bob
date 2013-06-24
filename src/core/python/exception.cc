/**
 * @file core/python/exception.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds the core extension into Python. Please note that, for each
 * exception type you only need to declare once the translator. All other
 * modules will benefit from it automatically.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <Python.h>
#include <bob/core/Exception.h>
#include <bob/core/array_exception.h>
#include <bob/python/exception.h>

using namespace bob::python;

/**
 * This method is only useful to test exception throwing in Python code.
 */
static void throw_exception(void) {
  throw bob::core::Exception();
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

  register_exception_translator<bob::core::Exception>(PyExc_RuntimeError);
  register_exception_translator<bob::core::NotImplementedError>(PyExc_NotImplementedError);
  register_exception_translator<bob::core::InvalidArgumentException>(PyExc_ValueError);

  // note: only register exceptions to which you need specific behavior not
  // covered by catching RuntimeError

  // just for tests...
  boost::python::def("throw_exception", &throw_exception);
}
