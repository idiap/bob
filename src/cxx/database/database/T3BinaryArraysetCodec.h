/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Converts data from the Torch3vision (bindata) format into Torch5spro. 
 */

#ifndef TORCH_DATABASE_TORCH3VSIONARRAYSETCODEC_H 
#define TORCH_DATABASE_TORCH3VSIONARRAYSETCODEC_H

#include "database/ArraysetCodec.h"

namespace Torch { namespace database {

  /**
   * T3BinaryArraysetCodecs can read and write single arrays into a
   * Torch-compatible binary file.
   */
  class T3BinaryArraysetCodec : public ArraysetCodec {

    public:

      T3BinaryArraysetCodec(bool save_in_float32=true);

      virtual ~T3BinaryArraysetCodec();

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
       */
      virtual void append(const std::string& filename, const Array& array) const;

      /**
       * Saves a representation of the given arrayset in the file.
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
      bool m_float; ///< should I save output as floats?

  };

}}

#endif /* TORCH_DATABASE_TORCH3VSIONARRAYSETCODEC_H */
