/**
 * @file io/ExternalArrayImpl.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief This class describes how an array will load and save data from an
 * external file.
 */

#ifndef TORCH_IO_EXTERNALARRAYIMPL_H 
#define TORCH_IO_EXTERNALARRAYIMPL_H

#include <string>

#include <blitz/array.h>
#include <boost/shared_ptr.hpp>

#include "core/array_type.h"

#include "io/InlinedArrayImpl.h"
#include "io/ArrayCodec.h"

namespace Torch { namespace io { namespace detail {

  /** 
   * An ExternalArrayImpl represents an array to which the data is stored on
   * an external file. It has no notion of management aspects of arrays, for
   * example, it does not hold an id or is related to a parent arrayset. It
   * just knows how to handle its data. Reading and writing data to an
   * ExternalArrayImpl will read and write data to/from a file. This operation
   * can be costly.
   */
  class ExternalArrayImpl {

    public:

      /**
       * Specify the filename and the code to read it. If you don't specify the
       * codec, I'll try to guess it from the registry. If the user gives a
       * relative path, it is relative to the current working directory. The
       * last parameter should be used in the case this is supposed to create a
       * new file, in which case the specifications will not be pre-loaded.
       */
      ExternalArrayImpl(const std::string& filename, const std::string&
          codecname="", bool newfile=false);

      /**
       * Destroys an array.
       */
      virtual ~ExternalArrayImpl();

      /**
       * Loads the data from the file
       */
      inline InlinedArrayImpl get() const { return m_codec->load(m_filename); }

      /**
       * Saves data in the file. Please note that blitz::Array<>'s will be
       * implicetly converted to InlinedArrayImpl as required.
       */
      void set(const InlinedArrayImpl& data);

      /**
       * Sets the filename of where the data is contained, re-writing the data
       * if the codec changes. Otherwise, just move.
       */
      void move(const std::string& filename, const std::string& codecname="");

      /**
       * Gets the filename
       */
      inline const std::string& getFilename() const { return m_filename; }

      /**
       * Gets a pointer to the codec.
       */
      inline boost::shared_ptr<const Torch::io::ArrayCodec> getCodec() const 
      { return m_codec; }

      /**
       * Some informative methods
       */
      inline Torch::core::array::ElementType getElementType() const { return m_elementtype; }
      inline size_t getNDim() const { return m_ndim; }
      inline const size_t* getShape() const { return m_shape; }

    private: //some helpers

      void reloadSpecification();

    private: //representation
      std::string m_filename; ///< The file where this array is stored
      boost::shared_ptr<const Torch::io::ArrayCodec> m_codec; ///< How to load and save the data
      Torch::core::array::ElementType m_elementtype;
      size_t m_ndim;
      size_t m_shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
  };

}}}

#endif /* TORCH_IO_EXTERNALARRAYIMPL_H */
