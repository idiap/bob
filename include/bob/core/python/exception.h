/**
 * @file bob/core/python/exception.h
 * @date Fri Mar 25 15:21:36 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements a few classes that are useful for binding bob exceptions
 * to python.
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

#ifndef BOB_PYTHON_CORE_EXCEPTION_H 
#define BOB_PYTHON_CORE_EXCEPTION_H

#include <boost/python.hpp>

/**
 * Raises a python exception with a formatted message
 */
#define PYTHON_ERROR(TYPE, ...) \
{ \
  PyErr_Format(PyExc_##TYPE, __VA_ARGS__); \
  throw boost::python::error_already_set(); \
}

/**
 * Raises a python warning with a formatted message
 */
#define PYTHON_WARNING(TYPE, MESSAGE) \
{ \
  PyErr_Warn(PyExc_##TYPE, MESSAGE); \
}

namespace bob { namespace python {

  /**
   * This is a generalized exception translator for boost python. It simplifies
   * translation declaration for as long as you provide a what() method in your
   * exception classes that return a const char* with the exception
   * description.
   *
   * If you follow that protocol, you should be able to do something like:
   *
   * ExceptionTranslator<std::out_of_range> t(PyExc_RuntimeError)
   *
   * On your boost::python modules.
   */
  template <typename T> struct ExceptionTranslator {

    public:

      void operator()(const T& cxx_except) const {
        PyErr_SetString(m_py_except, cxx_except.what());
      }

      ExceptionTranslator(PyObject* py_except): m_py_except(py_except) {
        boost::python::register_exception_translator<T>(*this);
      }

      ExceptionTranslator(const ExceptionTranslator& other):
        m_py_except(other.m_py_except) {
          //do not re-register the translator here!
      }

    private:

      PyObject* m_py_except;

  };

  /**
   * A thin wrapper to call the translator and escape the variable naming
   * issue when declaring multiple ExceptionTranslator's on the same module. 
   *
   * If you think about it, it would have to look like this:
   *
   * ExceptionTranslator<MyException1> translator1(PyExc_RuntimeError);
   * ExceptionTranslator<MyException2> translator2(PyExc_RuntimeError);
   *
   * Using this method will make it look like this:
   *
   * register_exception_translator<MyException1>(PyExc_RuntimeError);
   */
  template <typename T> void register_exception_translator(PyObject* e) {
    ExceptionTranslator<T> my_translator(e);
  }

}}

#endif /* BOB_PYTHON_CORE_EXCEPTION_H */
