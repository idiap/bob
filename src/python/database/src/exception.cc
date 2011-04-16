/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 26 Jan 14:29:39 2011 
 *
 * @brief Database exceptions 
 */

#include "database/Exception.h"
#include "core/python/exception.h"

using namespace Torch::core::python;

void bind_database_exception() {
  CxxToPythonTranslator<Torch::database::Exception, Torch::core::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");

  CxxToPythonTranslator<Torch::database::NonExistingElement, Torch::database::Exception>("NonExistingElement", "Raised when database elements types are not implemented");

  CxxToPythonTranslatorPar<Torch::database::IndexError, Torch::database::Exception, size_t>("IndexError", "Raised when database elements queried-for (addressable by id) do not exist");

  CxxToPythonTranslatorPar<Torch::database::NameError, Torch::database::Exception, const std::string&>("NameError", "Raised when database elements queried-for (addressable by name) do not exist");

  CxxToPythonTranslatorPar2<Torch::database::DimensionError, Torch::database::Exception, size_t, size_t>("DimensionError", "Raised when user asks for arrays with unsupported dimensionality");

  CxxToPythonTranslatorPar2<Torch::database::TypeError, Torch::database::Exception, Torch::core::array::ElementType, Torch::core::array::ElementType>("TypeError", "Raised when the user asks for arrays with unsupported element type");

  CxxToPythonTranslatorPar<Torch::database::UnsupportedTypeError, Torch::database::Exception, Torch::core::array::ElementType>("UnsupportedTypeError", "Raised when the user wants to performe an operation for which this particular type is not supported");

  CxxToPythonTranslator<Torch::database::Uninitialized, Torch::database::Exception>("Uninitialized", "Raised when the user asks for arrays with unsupported element type");

  CxxToPythonTranslatorPar<Torch::database::AlreadyHasRelations, Torch::database::Exception, size_t>("AlreadyHasRelations", "Raised when the user inserts a new rule to a Relationset with existing relations");

  CxxToPythonTranslator<Torch::database::InvalidRelation, Torch::database::Exception>("InvalidRelation", "Raised when the user inserts a new Relation to a Relationset that does not conform to its rules");
  
  CxxToPythonTranslatorPar<Torch::database::FileNotReadable, Torch::database::Exception, const std::string&>("FileNotReadable", "Raised when a file is not found or readable");

  CxxToPythonTranslatorPar<Torch::database::ExtensionNotRegistered, Torch::database::Exception, const std::string&>("ExtensionNotRegistered", "Raised when Codec Registry lookups by extension do not find a codec match for the given string");

  CxxToPythonTranslatorPar<Torch::database::CodecNotFound, Torch::database::Exception, const std::string&>("CodecNotFound", "Raised when the codec is looked-up by name and is not found");

  CxxToPythonTranslatorPar<Torch::database::PathIsNotAbsolute, Torch::database::Exception, const std::string&>("PathIsNotAbsolute", "Raised when an absolute path is required and the user fails to comply");

  CxxToPythonTranslatorPar<Torch::database::ImageUnsupportedDimension, Torch::database::Exception, const size_t>("ImageUnsupportedDimension", "Raised when an image has not a valid number of dimensions (2 for grayscale and 3 for RGB)");

  CxxToPythonTranslatorPar<Torch::database::ImageUnsupportedType, Torch::database::Exception, Torch::core::array::ElementType>("ImageUnsupportedType", "Raised when an image has not a valid type (uint8_t or uint16_t)");

  CxxToPythonTranslatorPar<Torch::database::ImageUnsupportedDepth, Torch::database::Exception, const unsigned int>("ImageUnsupportedDepth", "Raised when an image has not a valid depth (up to 16 bits depth images are supported)");
}
