/**
 * @file io/HDF5ArrayCodec.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Describes a generic API for reading and writing data to external
 * files.
 */

#ifndef TORCH_IO_HDF5ARRAYCODEC_H 
#define TORCH_IO_HDF5ARRAYCODEC_H

#include "io/ArrayCodec.h"

namespace Torch { namespace io {

  /**
   * HDF5ArrayCodecs can read and write single arrays into a Torch-compatible
   * binary file.
   */
  class HDF5ArrayCodec : public ArrayCodec {

    public:

      HDF5ArrayCodec();

      virtual ~HDF5ArrayCodec();

      /**
       * Returns the total amount of storage required to load the array into
       * memory, in number of elements of type 'dtype' (**not** the number of
       * bytes). Fills up array properties such as the element type, number of
       * dimensions, the shape and strides. The shape and strides use as basis
       * the sizeof(dtype) and **not** the number of bytes.
       */
      virtual void peek(const std::string& file, typeinfo& info) const;

      /**
       * Loads the array data into a pre-defined memory area. This memory area
       * has to have enough space as can be verified by "peek".
       *
       * This method will check to see if the given array has enough space. If
       * that is not the case, it will allocated enough space internally by
       * reseting the input array and putting the data read from the file
       * inside.
       */
      virtual void load(const std::string& file, buffer& array) const;

      /**
       * Saves a representation of the given array in the file and according to
       * the specifications defined on the interface.
       */
      virtual void save (const std::string& file, const buffer& array) const;

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

#endif /* TORCH_IO_HDF5ARRAYCODEC_H */
