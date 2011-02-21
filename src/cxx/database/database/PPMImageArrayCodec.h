/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Converts data from the Portable Pixelmap (PPM) P6 image format into
 * Torch5spro.
 */

#ifndef TORCH5SPRO_DATABASE_PPMIMAGE_ARRAYCODEC_H 
#define TORCH5SPRO_DATABASE_PPMIMAGE_ARRAYCODEC_H 1

#include "database/ArrayCodec.h"

namespace Torch { namespace database {

  /**
   * Reads and writes single Arrays to PPM P6 image files
   */
  class PPMImageArrayCodec : public ArrayCodec {

    public:

      PPMImageArrayCodec();

      virtual ~PPMImageArrayCodec();

      /**
       * Returns the element type and the number of dimensions of the stored
       * array.
       */
      virtual void peek(const std::string& filename, 
          Torch::core::array::ElementType& eltype, size_t& ndim,
          size_t* shape) const;

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
      /**
`       * Parses the header of the file and return the width and the height
        * An exception is thrown if the header is not in PGM format.
        */
      void 
      parseHeader(std::ifstream& ifile, size_t& height, size_t& width) const;

      std::string m_name; ///< my own name
      std::vector<std::string> m_extensions; ///< extensions I can handle

  };

}}

#endif /* TORCH5SPRO_DATABASE_PGMIMAGE_ARRAYCODEC_H */
