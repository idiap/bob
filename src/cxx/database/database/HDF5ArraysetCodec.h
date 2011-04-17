/**
 * @file database/HDF5ArraysetCodec.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Describes a generic API for reading and writing data to external
 * files.
 */

#ifndef TORCH_DATABASE_HDF5ARRAYSETCODEC_H 
#define TORCH_DATABASE_HDF5ARRAYSETCODEC_H

#include "database/ArraysetCodec.h"

namespace Torch { namespace database {

  /**
   * HDF5ArraysetCodecs can read and write single arrays into a
   * Torch-compatible binary file.
   */
  class HDF5ArraysetCodec : public ArraysetCodec {

    public:

      HDF5ArraysetCodec();

      virtual ~HDF5ArraysetCodec();

      /**
       * Returns the element type and the number of dimensions of the stored
       * array.
       */
      virtual void peek(const std::string& filename, 
          Torch::core::array::ElementType& eltype, size_t& ndim,
          size_t* shape, size_t& samples) const;

      /**
       * Returns the fully stored Arrayset in a InlinedArraysetImpl
       */
      virtual detail::InlinedArraysetImpl load(const std::string& filename) const;

      /**
       * Loads a single Array from the array set. Please note that this may
       * raise exceptions in case the index positioning in the file exceeds
       * the number of arrays saved in that file.
       */
      virtual Array load(const std::string& filename, size_t index) const;

      /**
       * Appends a new Array in the file. Please note that there may be
       * restrictions on which kinds of arrays each file type can receive.
       *
       * @warning: Please convert your files to HDF5, this codec is deprecated
       * starting on 16.04.2011 - AA
       */
      virtual void append(const std::string& filename, const Array& array) const;

      /**
       * Saves a representation of the given arrayset in the file.
       *
       * @warning: Please convert your files to HDF5, this codec is deprecated
       * starting on 16.04.2011 - AA
       */
      virtual void save (const std::string& filename, 
          const detail::InlinedArraysetImpl& data) const;

      /**
       * Returns the name of this codec
       */
      virtual inline const std::string& name () const { return m_name; }

      /**
       * Returns a list of known extensions this codec can handle. The
       * extensions include the initial ".". So, to cover for jpeg images, you
       * may return a vector containing ".jpeg" and ".jpg" for example. Case
       * matters, so ".jpeg" and ".JPEG" are different extensions. If are the
       * responsible to cover all possible variations an extension can have.
       */
      virtual inline const std::vector<std::string>& extensions () const { return m_extensions; }

    private: //representation

      std::string m_name; ///< my own name
      std::vector<std::string> m_extensions; ///< extensions I can handle

  };

}}

#endif /* TORCH_DATABASE_HDF5ARRAYSETCODEC_H */
