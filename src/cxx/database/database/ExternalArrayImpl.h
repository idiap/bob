/**
 * @file database/ExternalArrayImpl.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief This class describes how an array will load and save data from an
 * external file.
 */

#ifndef TORCH_DATABASE_EXTERNALARRAYIMPL_H 
#define TORCH_DATABASE_EXTERNALARRAYIMPL_H

#include <string>

#include <blitz/array.h>
#include <boost/shared_ptr.hpp>

#include "core/array_common.h"
#include "database/ArrayCodec.h"

namespace Torch { namespace database { namespace detail {

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
       * relative path, it is relative to the current working directory.
       */
      ExternalArrayImpl(const std::string& filename, const std::string&
          codecname="");

      /**
       * Destroys an array.
       */
      virtual ~ExternalArrayImpl();

      /**
       * Returns a new array from the data in the file.
       */
      template<typename T, int D> inline blitz::Array<T,D> load() const {
        return load().cast<T,D>();
      }

      /**
       * Loads the data from the file
       */
      inline void InlinedArrayImpl load() const {
        return m_codec->load(m_filename);
      }

      /**
       * Saves data in the file. Please note that blitz::Array<>'s will be
       * implicetly converted to InlinedArrayImpl as required.
       */
      inline void save(const InlinedArrayImpl& data) {
        m_codec->save(m_filename, data);
      }

      /**
       * Returns the specifications of the array contained in the file. This
       * operation is delegated to the codec that may open the file and read
       * values from the file to return you sensible data.
       *
       * TODO: We could try to optimize this call by caching the element type
       * and number of dimensions after first reading them. Please note you
       * have to solve the "empty file" or "non-existant file" problem for this
       * to work reliably.
       */
      void getSpecification(Torch::core::array::ElementType& eltype,
          size_t& ndim, 
          size_t& shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY]) const;

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
      inline boost::shared_ptr<const Torch::database::ArrayCodec> getCodec() const 
      { return m_codec; }

    private: //representation
      std::string m_filename; ///< The file where this array is stored
      boost::shared_ptr<const Torch::database::ArrayCodec> m_codec; ///< How to load and save the data
  };

}}}

#endif /* TORCH_DATABASE_EXTERNALARRAYIMPL_H */
