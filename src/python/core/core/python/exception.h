/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 25 Mar 15:05:55 2011 
 *
 * @brief Implements a few classes that are useful for binding Torch exceptions
 * to python.
 */

#ifndef TORCH_PYTHON_CORE_EXCEPTION_H 
#define TORCH_PYTHON_CORE_EXCEPTION_H

#include <boost/python.hpp>

/**
 * Raises a python exception with a formatted message
 */
#define PYTHON_ERROR(TYPE, ...) \
{ \
  PyErr_Format(PyExc_##TYPE, __VA_ARGS__); \
  throw boost::python::error_already_set(); \
}

namespace Torch { namespace python {

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

#endif /* TORCH_PYTHON_CORE_EXCEPTION_H */
