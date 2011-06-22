/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 26 Jan 14:29:39 2011 
 *
 * @brief io exceptions 
 */

#include "io/Exception.h"
#include "core/python/exception.h"

using namespace Torch::core::python;

void bind_io_exception() {
  CxxToPythonTranslator<Torch::io::Exception, Torch::core::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");

  CxxToPythonTranslator<Torch::io::NonExistingElement, Torch::io::Exception>("NonExistingElement", "Raised when io elements types are not implemented");

  CxxToPythonTranslatorPar<Torch::io::IndexError, Torch::io::Exception, size_t>("IndexError", "Raised when io elements queried-for (addressable by id) do not exist");

  CxxToPythonTranslatorPar<Torch::io::NameError, Torch::io::Exception, const std::string&>("NameError", "Raised when io elements queried-for (addressable by name) do not exist");

  CxxToPythonTranslatorPar2<Torch::io::DimensionError, Torch::io::Exception, size_t, size_t>("DimensionError", "Raised when user asks for arrays with unsupported dimensionality");

  CxxToPythonTranslatorPar2<Torch::io::TypeError, Torch::io::Exception, Torch::core::array::ElementType, Torch::core::array::ElementType>("TypeError", "Raised when the user asks for arrays with unsupported element type");

  CxxToPythonTranslatorPar<Torch::io::UnsupportedTypeError, Torch::io::Exception, Torch::core::array::ElementType>("UnsupportedTypeError", "Raised when the user wants to performe an operation for which this particular type is not supported");

  CxxToPythonTranslator<Torch::io::Uninitialized, Torch::io::Exception>("Uninitialized", "Raised when the user asks for arrays with unsupported element type");

  CxxToPythonTranslatorPar<Torch::io::FileNotReadable, Torch::io::Exception, const std::string&>("FileNotReadable", "Raised when a file is not found or readable");

  CxxToPythonTranslatorPar<Torch::io::ExtensionNotRegistered, Torch::io::Exception, const std::string&>("ExtensionNotRegistered", "Raised when Codec Registry lookups by extension do not find a codec match for the given string");

  CxxToPythonTranslatorPar<Torch::io::CodecNotFound, Torch::io::Exception, const std::string&>("CodecNotFound", "Raised when the codec is looked-up by name and is not found");

  CxxToPythonTranslatorPar<Torch::io::PathIsNotAbsolute, Torch::io::Exception, const std::string&>("PathIsNotAbsolute", "Raised when an absolute path is required and the user fails to comply");

  CxxToPythonTranslatorPar<Torch::io::ImageUnsupportedDimension, Torch::io::Exception, const size_t>("ImageUnsupportedDimension", "Raised when an image has not a valid number of dimensions (2 for grayscale and 3 for RGB)");

  CxxToPythonTranslatorPar<Torch::io::ImageUnsupportedType, Torch::io::Exception, Torch::core::array::ElementType>("ImageUnsupportedType", "Raised when an image has not a valid type (uint8_t or uint16_t)");

  CxxToPythonTranslatorPar<Torch::io::ImageUnsupportedDepth, Torch::io::Exception, const unsigned int>("ImageUnsupportedDepth", "Raised when an image has not a valid depth (up to 16 bits depth images are supported)");
}
