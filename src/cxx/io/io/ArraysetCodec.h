/**
 * @file io/ArrayCodec.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Describes a generic API for reading and writing data to external
 * files.
 */

#ifndef TORCH_IO_ARRAYSETCODEC_H 
#define TORCH_IO_ARRAYSETCODEC_H

#include <vector>
#include <string>

#include "io/InlinedArraysetImpl.h"

namespace Torch { namespace io {

  /**
   * ArraysetCodecs are sets of types that can deal with external files, reading
   * and writing a multiple arrays to and from them. This is a pure virtual
   * type. You have to subclass it and register your codec with our
   * ArraysetCodecRegistry singleton to make it work transparently.
   */
  class ArraysetCodec {

    public:

      virtual ~ArraysetCodec();

      /**
       * Returns the element type and the number of dimensions of the stored
       * array.
       */
      virtual void peek(const std::string& filename, 
          Torch::core::array::ElementType& eltype, size_t& ndim,
          size_t* shape, size_t& samples) const =0;

      /**
       * Returns the fully stored Arrayset in a InlinedArraysetImpl
       */
      virtual detail::InlinedArraysetImpl load(const std::string& filename) const =0;

      /**
       * Loads a single Array from the array set. Please note that this may
       * raise exceptions in case the index positioning in the file exceeds
       * the number of arrays saved in that file.
       */
      virtual Array load(const std::string& filename, size_t index) const =0;

      /**
       * Appends a new Array in the file. Please note that there may be
       * restrictions on which kinds of arrays each file type can receive.
       */
      virtual void append(const std::string& filename, const Array& array) const =0;

      /**
       * Saves a representation of the given arrayset in the file.
       */
      virtual void save (const std::string& filename, 
          const detail::InlinedArraysetImpl& data) const =0;

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

#endif /* TORCH_IO_ARRAYSETCODEC_H */
