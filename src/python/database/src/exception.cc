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
