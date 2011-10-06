/**
 * @file io/ArrayCodec.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Describes a generic API for reading and writing data to external
 * files.
 */

#ifndef TORCH_IO_ARRAYCODEC_H 
#define TORCH_IO_ARRAYCODEC_H

#include <vector>
#include <string>
#include "io/buffer.h"

namespace Torch { namespace io {

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
       * Peeks the type of array stored inside the file
       */
      virtual void peek(const std::string& file, typeinfo& info) const =0;

      /**
       * Loads the data of the array into memory.
       *
       * This method will check to see if the given array has enough space. If
       * that is not the case, it will allocate enough space internally by
       * reseting the input array and putting the data read from the file
       * inside.
       */
      virtual void load(const std::string& file, buffer& buffer) const =0;

      /**
       * Saves a representation of the given array in the file and according to
       * the specifications defined on the interface.
       */
      virtual void save (const std::string& file, const buffer& buffer) const =0;

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

#endif /* TORCH_IO_ARRAYCODEC_H */
