/**
 * @file database/ArrayCodec.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Describes a generic API for reading and writing data to external
 * files.
 */

#ifndef TORCH_DATABASE_ARRAYCODEC_H 
#define TORCH_DATABASE_ARRAYCODEC_H

#include <vector>
#include <string>

#include "database/InlinedArrayImpl.h"

namespace Torch { namespace database {

  /**
   * ArrayCodecs are sets of types that can deal with external files, reading
   * and writing a <b>single</b> to and from them. This is a pure virtual type.
   * You have to subclass it and register your codec with our
   * ArrayCodecRegistry singleton to make it work transparently.
   */
  class ArrayCodec {

    public:

      virtual ~ArrayCodec();

      /**
       * Returns the element type and the number of dimensions of the stored
       * array.
       */
      virtual void peek(const std::string& filename, 
          Torch::core::array::ElementType& eltype, size_t& ndim,
          size_t* shape) const =0;

      /**
       * Returns the stored array in a InlinedArrayImpl
       */
      virtual detail::InlinedArrayImpl load(const std::string& filename) const =0;

      /**
       * Saves a representation of the given array in the file.
       */
      virtual void save (const std::string& filename, 
          const detail::InlinedArrayImpl& data) const =0;

      /**
       * Returns the name of this codec
       */
      virtual const std::string& name () const =0;

      /**
       * Returns a list of known extensions this codec can handle. The
       * extensions include the initial ".". So, to cover for jpeg images, you
       * may return a vector containing ".jpeg" and ".jpg" for example. Case
       * matters, so ".jpeg" and ".JPEG" are different extensions. If are the
       * responsible to cover all possible variations an extension can have.
       */
      virtual const std::vector<std::string>& extensions () const =0;

  };

}}

#endif /* TORCH_DATABASE_ARRAYCODEC_H */
