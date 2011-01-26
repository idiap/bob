/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 26 Jan 14:29:39 2011 
 *
 * @brief Database exceptions 
 */

#include <boost/python.hpp>

#include "core/Dataset2.h"

using namespace boost::python;

/**
 * The following lines of code implement exception translation from C++ into
 * python using Boost.Python and the instructions found on this webpage:
 *
 * http://stackoverflow.com/questions/2261858/boostpython-export-custom-exception
 * They were just slightly modified to make it easier to apply the code for
 * different situations.
 */

template <typename T> struct CxxToPythonTranslator {
  /**
   * This static class variable will hold a pointer to the exception type as
   * defined by the boost::python
   */
  static PyObject* pyExceptionType;

  /**
   * Do the exception translation for the specific exception we are trying to
   * tackle.
   */
  static void translateException(const T& ex) {
    assert(pyExceptionType != NULL);
    boost::python::object pythonExceptionInstance(ex);
    PyErr_SetObject(pyExceptionType, pythonExceptionInstance.ptr());
  }

  /**
   * Constructor will instantiate all required parameters for this standard
   * exception handler and create the pythonic bindings in one method call
   */
  CxxToPythonTranslator(const char* python_name, const char* python_doc) {
    class_<T> pythonEquivalentException(python_name, python_doc, init<>("Creates a new exception of this type"));
    pythonEquivalentException.def("__str__", &T::what);
    pyExceptionType = pythonEquivalentException.ptr();
    register_exception_translator<T>(&translateException);
  }

};

template <typename T> PyObject* CxxToPythonTranslator<T>::pyExceptionType = 0;

#define BIND_EXCEPTION(TYPE,NAME,DOC) CxxToPythonTranslator<TYPE>(NAME, DOC)

void bind_database_exception() {
  BIND_EXCEPTION(Torch::core::NonExistingElement, "NonExistingElement", "Raised when database elements that were queried for do not exist");
  BIND_EXCEPTION(Torch::core::IndexError, "IndexError", "Raised when database elements queried-for do not exist");
  BIND_EXCEPTION(Torch::core::NDimensionError, "NDimensionError", "Raised when user asks for arrays with unsupported dimensionality");
  BIND_EXCEPTION(Torch::core::TypeError, "TypeError", "Raised when user asks for arrays with unsupported element type");
}
