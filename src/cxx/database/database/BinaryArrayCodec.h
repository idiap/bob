/**
 * @file database/BinaryArrayCodec.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Describes a generic API for reading and writing data to external
 * files.
 */

#ifndef TORCH_DATABASE_BINARYARRAYCODEC_H 
#define TORCH_DATABASE_BINARYARRAYCODEC_H

#include "database/ArrayCodec.h"

namespace Torch { namespace database {

  /**
   * BinaryArrayCodecs can read and write single arrays into a Torch-compatible
   * binary file.
   */
  class BinaryArrayCodec {

    public:

      BinaryArrayCodec();

      virtual ~BinaryArrayCodec();

      /**
       * Returns the element type and the number of dimensions of the stored
       * array.
       */
      virtual void peek(const std::string& filename, 
          Torch::core::array::ElementType& eltype, size_t& ndim) const;

      /**
       * Returns the stored array in a InlinedArrayImpl
       */
      virtual detail::InlinedArrayImpl load(const std::string& filename) const;

      /**
       * Saves a representation of the given array in the file.
       */
      virtual void save (const std::string& filename, 
          const detail::InlinedArrayImpl& data) const;

      /**
       * Returns the name of this codec
       */
      virtual const std::string& name () const;

      /**
       * Returns a list of known extensions this codec can handle. The
       * extensions include the initial ".". So, to cover for jpeg images, you
       * may return a vector containing ".jpeg" and ".jpg" for example. Case
       * matters, so ".jpeg" and ".JPEG" are different extensions. If are the
       * responsible to cover all possible variations an extension can have.
       */
      virtual const std::vector<std::string>& extensions () const;

    private: //representation

      std::string m_name; ///< my own name
      std::vector<std::string> m_extensions; ///< extensions I can handle

  };

}}

#endif /* TORCH_DATABASE_BINARYARRAYCODEC_H */
