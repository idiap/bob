/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 26 Jan 14:29:39 2011 
 *
 * @brief Database exceptions 
 */

#include <boost/python.hpp>

#include "database/Exception.h"

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

template <typename T, typename TPar> struct CxxToPythonTranslatorPar {
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
  CxxToPythonTranslatorPar(const char* python_name, const char* python_doc) {
    class_<T> pythonEquivalentException(python_name, python_doc, init<TPar>("Creates a new exception of this type"));
    pythonEquivalentException.def("__str__", &T::what);
    pyExceptionType = pythonEquivalentException.ptr();
    register_exception_translator<T>(&translateException);
  }

};

template <typename T, typename TPar1, typename TPar2> struct CxxToPythonTranslatorPar2 {
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
  CxxToPythonTranslatorPar2(const char* python_name, const char* python_doc) {
    class_<T> pythonEquivalentException(python_name, python_doc, init<TPar1, TPar2>("Creates a new exception of this type"));
    pythonEquivalentException.def("__str__", &T::what);
    pyExceptionType = pythonEquivalentException.ptr();
    register_exception_translator<T>(&translateException);
  }

};

template <typename T> PyObject* CxxToPythonTranslator<T>::pyExceptionType = 0;
template <typename T, typename TPar> PyObject* CxxToPythonTranslatorPar<T,TPar>::pyExceptionType = 0;
template <typename T, typename TPar1, typename TPar2> PyObject* CxxToPythonTranslatorPar2<T,TPar1,TPar2>::pyExceptionType = 0;

void bind_database_exception() {
  CxxToPythonTranslator<Torch::database::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");

  CxxToPythonTranslator<Torch::database::NonExistingElement>("NonExistingElement", "Raised when database elements types are not implemented");

  CxxToPythonTranslatorPar<Torch::database::IndexError, size_t>("IndexError", "Raised when database elements queried-for (addressable by id) do not exist");

  CxxToPythonTranslatorPar<Torch::database::NameError, const std::string&>("NameError", "Raised when database elements queried-for (addressable by name) do not exist");

  CxxToPythonTranslatorPar2<Torch::database::DimensionError, size_t, size_t>("DimensionError", "Raised when user asks for arrays with unsupported dimensionality");

  CxxToPythonTranslatorPar2<Torch::database::TypeError, Torch::core::array::ElementType, Torch::core::array::ElementType>("TypeError", "Raised when the user asks for arrays with unsupported element type");

  CxxToPythonTranslatorPar<Torch::database::UnsupportedTypeError, Torch::core::array::ElementType>("UnsupportedTypeError", "Raised when the user wants to performe an operation for which this particular type is not supported");

  CxxToPythonTranslator<Torch::database::Uninitialized>("Uninitialized", "Raised when the user asks for arrays with unsupported element type");

  CxxToPythonTranslatorPar<Torch::database::AlreadyHasRelations, size_t>("AlreadyHasRelations", "Raised when the user inserts a new rule to a Relationset with existing relations");

  CxxToPythonTranslator<Torch::database::InvalidRelation>("InvalidRelation", "Raised when the user inserts a new Relation to a Relationset that does not conform to its rules");
  
  CxxToPythonTranslatorPar<Torch::database::FileNotReadable, const std::string&>("FileNotReadable", "Raised when a file is not found or readable");

  CxxToPythonTranslatorPar<Torch::database::ExtensionNotRegistered, const std::string&>("ExtensionNotRegistered", "Raised when Codec Registry lookups by extension do not find a codec match for the given string");

  CxxToPythonTranslatorPar<Torch::database::CodecNotFound, const std::string&>("CodecNotFound", "Raised when the codec is looked-up by name and is not found");

  CxxToPythonTranslatorPar<Torch::database::PathIsNotAbsolute, const std::string&>("PathIsNotAbsolute", "Raised when an absolute path is required and the user fails to comply");
}
