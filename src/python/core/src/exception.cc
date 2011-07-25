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

using namespace Torch::core::python;
using namespace boost::python;

/**
 * This method is only useful to test exception throwing in Python code.
 */
static void throw_exception(void) {
  throw Torch::core::Exception();
}

/**
 * PYTHON EXCEPTION HIERARCHY:
 * ---------------------------
 *
 * BaseException
 *  +-- SystemExit
 *  +-- KeyboardInterrupt
 *  +-- GeneratorExit
 *  +-- Exception
 *       +-- StopIteration
 *       +-- StandardError
 *       |    +-- BufferError
 *       |    +-- ArithmeticError
 *       |    |    +-- FloatingPointError
 *       |    |    +-- OverflowError
 *       |    |    +-- ZeroDivisionError
 *       |    +-- AssertionError
 *       |    +-- AttributeError
 *       |    +-- EnvironmentError
 *       |    |    +-- IOError
 *       |    |    +-- OSError
 *       |    |         +-- WindowsError (Windows)
 *       |    |         +-- VMSError (VMS)
 *       |    +-- EOFError
 *       |    +-- ImportError
 *       |    +-- LookupError
 *       |    |    +-- IndexError
 *       |    |    +-- KeyError
 *       |    +-- MemoryError
 *       |    +-- NameError
 *       |    |    +-- UnboundLocalError
 *       |    +-- ReferenceError
 *       |    +-- RuntimeError
 *       |    |    +-- NotImplementedError
 *       |    +-- SyntaxError
 *       |    |    +-- IndentationError
 *       |    |         +-- TabError
 *       |    +-- SystemError
 *       |    +-- TypeError
 *       |    +-- ValueError
 *       |         +-- UnicodeError
 *       |              +-- UnicodeDecodeError
 *       |              +-- UnicodeEncodeError
 *       |              +-- UnicodeTranslateError
 */

// Here, a few bindings to standard exceptions.
void translator_except(std::exception const& x) { 
  PYTHON_ERROR(Exception, x.what()); 
}

void translator_logic_error(std::logic_error const& x) { 
  PYTHON_ERROR(RuntimeError, x.what()); 
}

void translator_domain_error(std::domain_error const& x) { 
  PYTHON_ERROR(RuntimeError, x.what()); 
}

void translator_invalid_argument(std::invalid_argument const& x) { 
  PYTHON_ERROR(RuntimeError, x.what()); 
}

void translator_length_error(std::length_error const& x) { 
  PYTHON_ERROR(IndexError, x.what()); 
}

void translator_out_of_range(std::out_of_range const& x) { 
  PYTHON_ERROR(IndexError, x.what()); 
}

void translator_runtime_error(std::runtime_error const& x) { 
  PYTHON_ERROR(RuntimeError, x.what()); 
}

void translator_range_error(std::range_error const& x) { 
  PYTHON_ERROR(IndexError, x.what()); 
}

void translator_overflow_error(std::overflow_error const& x) { 
  PYTHON_ERROR(OverflowError, x.what()); 
}

void translator_underflow_error(std::underflow_error const& x) { 
  PYTHON_ERROR(ArithmeticError, x.what()); 
}

void bind_core_exception() {
  register_exception_translator<std::exception>(translator_except);
  register_exception_translator<std::logic_error>(translator_logic_error);
  register_exception_translator<std::domain_error>(translator_domain_error);
  register_exception_translator<std::invalid_argument>(translator_invalid_argument);
  register_exception_translator<std::length_error>(translator_length_error);
  register_exception_translator<std::out_of_range>(translator_out_of_range);
  register_exception_translator<std::runtime_error>(translator_runtime_error);
  register_exception_translator<std::range_error>(translator_range_error);
  register_exception_translator<std::overflow_error>(translator_overflow_error);
  register_exception_translator<std::underflow_error>(translator_underflow_error);

  BaseCxxToPythonTranslator<Torch::core::Exception>("Exception", "The core Exception class should be used as a basis for all Torch-Python exceptions.");
  CxxToPythonTranslatorPar<Torch::core::DeprecationError, Torch::core::Exception, const std::string&>("DeprecationError", "A deprecation error is raised when the developer wants to avoid the use of certain functionality in the code and for the user to migrate his code.");
  CxxToPythonTranslator<Torch::core::ConvertZeroInputRange, Torch::core::Exception>("ConvertZeroInputRange", "A ConvertZeroInputRange error is raised when the user try to convert an array which has a zero width input range.");
  CxxToPythonTranslatorPar2<Torch::core::ConvertInputAboveMaxRange, Torch::core::Exception, const double, const double>("ConvertInputAboveMaxRange", "A ConvertInputAboveMaxRange exception is raised when an input value is larger than the maximum of the input range given to the convert function.");
  CxxToPythonTranslatorPar2<Torch::core::ConvertInputBelowMinRange, Torch::core::Exception, const double, const double>("ConvertInputBelowMinRange", "A ConvertInputBelowMinRange exception is raised when an input value is smaller than the minimum of the input range given to the convert function.");
  CxxToPythonTranslatorPar2<Torch::core::NonZeroBaseError, Torch::core::Exception, const int, const int>("NonZeroBaseError", "The non-zero base error exception occurs when some function, which requires blitz Arrays to have zero base indices (for efficiency purpose), is used with an array which does not fulfill this requirement.");
  CxxToPythonTranslatorPar2<Torch::core::NonOneBaseError, Torch::core::Exception, const int, const int>("NonOneBaseError", "The non-one base error exception occurs when some function, which requires blitz Arrays to have one base indices (for efficiency purpose), is used with an array which does not fulfill this requirement.");
  CxxToPythonTranslator<Torch::core::NonCContiguousError, Torch::core::Exception>("NonCContiguousError", "The non-C contiguous error exception occurs when some function, which requires blitz Arrays to be stored contiguously in memory, is used with an array which does not fulfill this requirement.");
  CxxToPythonTranslator<Torch::core::NonFortranContiguousError, Torch::core::Exception>("NonFortranContiguousError", "The non-Fortran contiguous error exception occurs when some function, which requires blitz Arrays to be stored contiguously in memory, is used with an array which does not fulfill this requirement.");
  CxxToPythonTranslator<Torch::core::UnexpectedShapeError, Torch::core::Exception>("UnexpectedShapeError", "The UnexpectedShapeError exception occurs when a blitz array does not have the expected size.");
  CxxToPythonTranslator<Torch::core::DifferentBaseError, Torch::core::Exception>("DifferentBaseError", "The DifferentBaseError exception occurs when two blitz arrays do not have the same base indices, whereas this is required.");

  boost::python::def("throw_exception", &throw_exception);
}
