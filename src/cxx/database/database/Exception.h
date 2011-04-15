/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 16 Feb 13:42:04 2011 
 *
 * @brief Exceptions used throughout the Database subsystem of Torch
 */

#ifndef TORCH_DATABASE_EXCEPTION_H 
#define TORCH_DATABASE_EXCEPTION_H

#include <cstdlib>
#include "core/Exception.h"
#include "core/array_type.h"

namespace Torch { namespace database {

  class Exception: public Torch::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  class NameError: public Exception {
    public:
      NameError(const std::string& key) throw();
      virtual ~NameError() throw();
      virtual const char* what() const throw();

    private:
      std::string m_name;
      mutable std::string m_message;
  };

  class IndexError: public Exception {
    public:
      IndexError(size_t id) throw();
      virtual ~IndexError() throw();
      virtual const char* what() const throw();

    private:
      size_t m_id;
      mutable std::string m_message;
  };

  class DimensionError: public Exception {
    public:
      DimensionError(size_t got, size_t expected) throw();
      virtual ~DimensionError() throw();
      virtual const char* what() const throw();

    private:
      size_t m_got;
      size_t m_expected;
      mutable std::string m_message;
  };

  class TypeError: public Exception {
    public:
      TypeError(Torch::core::array::ElementType got, 
          Torch::core::array::ElementType expected) throw();
      virtual ~TypeError() throw();
      virtual const char* what() const throw();

    private:
      Torch::core::array::ElementType m_got;
      Torch::core::array::ElementType m_expected;
      mutable std::string m_message;
  };

  class UnsupportedTypeError: public Exception {
    public:
      UnsupportedTypeError(Torch::core::array::ElementType eltype) throw();
      virtual ~UnsupportedTypeError() throw();
      virtual const char* what() const throw();

    private:
      Torch::core::array::ElementType m_eltype;
      mutable std::string m_message;
  };

  class NonExistingElement: public Exception {
    public:
      NonExistingElement();
      virtual ~NonExistingElement() throw();
      virtual const char* what() const throw();
  };

  class Uninitialized: public Exception {
    public:
      Uninitialized() throw();
      virtual ~Uninitialized() throw();
      virtual const char* what() const throw();
  };

  class AlreadyHasRelations: public Exception {
    public:
      AlreadyHasRelations(size_t number) throw();
      virtual ~AlreadyHasRelations() throw();
      virtual const char* what() const throw();

    private:
      size_t m_number;
      mutable std::string m_message;
  };

  class InvalidRelation: public Exception {
    public:
      InvalidRelation() throw();
      virtual ~InvalidRelation() throw();
      virtual const char* what() const throw();
  };

  class FileNotReadable: public Exception {
    public:
      FileNotReadable(const std::string& filename) throw();
      virtual ~FileNotReadable() throw();
      virtual const char* what() const throw();

    private:
      std::string m_name;
      mutable std::string m_message;
  };

  class ExtensionNotRegistered: public Exception {
    public:
      ExtensionNotRegistered(const std::string& extension) throw();
      virtual ~ExtensionNotRegistered() throw();
      virtual const char* what() const throw();

    private:
      std::string m_extension;
      mutable std::string m_message;
  };

  class CodecNotFound: public Exception {
    public:
      CodecNotFound(const std::string& codecname) throw();
      virtual ~CodecNotFound() throw();
      virtual const char* what() const throw();

    private:
      std::string m_name;
      mutable std::string m_message;
  };

  class PathIsNotAbsolute: public Exception {
    public:
      PathIsNotAbsolute(const std::string& path) throw();
      virtual ~PathIsNotAbsolute() throw();
      virtual const char* what() const throw();

    private:
      std::string m_path;
      mutable std::string m_message;
  };

  class HDF5UnsupportedTypeError: public Exception {
    public:
      HDF5UnsupportedTypeError() throw();
      virtual ~HDF5UnsupportedTypeError() throw();
      virtual const char* what() const throw();

    private:
      mutable std::string m_message;
  };

  class HDF5UnsupportedDimensionError: public Exception {
    public:
      HDF5UnsupportedDimensionError(size_t n_dim) throw();
      virtual ~HDF5UnsupportedDimensionError() throw();
      virtual const char* what() const throw();

    private:
      size_t m_n_dim;
      mutable std::string m_message;
  };

  class HDF5ObjectNotFoundError: public Exception {
    public:
      HDF5ObjectNotFoundError(const std::string& path, const std::string& filename) throw();
      virtual ~HDF5ObjectNotFoundError() throw();
      virtual const char* what() const throw();

    private:
      std::string m_path;
      std::string m_filename;
      mutable std::string m_message;
  };

  class HDF5InvalidFileAccessModeError: public Exception {
    public:
      HDF5InvalidFileAccessModeError(const unsigned int mode) throw();
      virtual ~HDF5InvalidFileAccessModeError() throw();
      virtual const char* what() const throw();

    private:
      unsigned int m_mode;
      mutable std::string m_message;
  };

  class ImageUnsupportedDimension: public Exception {
    public:
      ImageUnsupportedDimension(const size_t n_dim) throw();
      virtual ~ImageUnsupportedDimension() throw();
      virtual const char* what() const throw();

    private:
      unsigned int m_n_dim;
      mutable std::string m_message;
  };
  
  class ImageUnsupportedType: public Exception {
    public:
      ImageUnsupportedType(const Torch::core::array::ElementType el_type)
        throw();
      virtual ~ImageUnsupportedType() throw();
      virtual const char* what() const throw();

    private:
      Torch::core::array::ElementType m_el_type;
      mutable std::string m_message;
  };

  class ImageUnsupportedDepth: public Exception {
    public:
      ImageUnsupportedDepth(const unsigned int depth) throw();
      virtual ~ImageUnsupportedDepth() throw();
      virtual const char* what() const throw();

    private:
      unsigned int m_depth;
      mutable std::string m_message;
  };

}}

#endif /* TORCH_DATABASE_EXCEPTION_H */
