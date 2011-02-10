/**
 * @file database/ExternalArraysetImpl.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief This class describes how an arrayset will load and save data from an
 * external file.
 */

#ifndef TORCH_DATABASE_EXTERNALARRAYSETIMPL_H 
#define TORCH_DATABASE_EXTERNALARRAYSETIMPL_H

#include <string>

#include <blitz/array.h>
#include <boost/shared_ptr.hpp>

#include "core/array_common.h"
#include "database/ArraysetCodec.h"

namespace Torch { namespace database { namespace detail {

  /** 
   * An ExternalArraysetImpl represents an arrayset to which the data is stored
   * on an external file. It has no notion of management aspects of arrays, for
   * example, it does not hold an id or is related to a parent arrayset. It
   * just knows how to handle its data. Reading and writing data to an
   * ExternalArraysetImpl will read and write data to/from a file. This
   * operation can be costly.
   */
  class ExternalArraysetImpl {

    public:

      /**
       * Specify the filename and the codec to read it. If you don't specify the
       * codec, I'll try to guess it from the registry. If the user gives a
       * relative path, it is relative to the current working directory.
       */
      ExternalArraysetImpl(const std::string& filename, const std::string&
          codecname="");

      /**
       * Destroys an arrayset.
       */
      virtual ~ExternalArraysetImpl();

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
      inline boost::shared_ptr<const Torch::database::ArraysetCodec> getCodec() const 
      { return m_codec; }

      /**
       * Accesses a single array by their id
       */
      Torch::database::Array operator[] (size_t id) const;

      /**
       * Appends a new array to the file I have. The 'id' field of such array
       * is ignored (even if it is non-zero). You cannot set the array 'id' in 
       * a file-based arrayset. Ids for arrays are given depending on their
       * position (1st array => id "1", 2nd array => id "2", and so on).
       */
      size_t add(boost::shared_ptr<const Torch::database::Array> array);
      size_t add(const Torch::database::Array& array);
      void extend(const InlinedArraysetImpl& set);

      /**
       * Removes a certain Array from this set. Please note this will trigger
       * loading and re-writing the underlying file. If the id index is out of
       * bounds, this is a noop. The numbering scheme for the ids is
       * "fortran-based", so we start counting at 1.
       */
      void remove(size_t id);
      void remove(boost::shared_ptr<const Torch::database::Array> array);
      void remove(const Torch::database::Array& array);

      /**
       * Loads the arrayset in memory in one shot.
       */
      InlinedArraysetImpl get() const;

      /**
       * Saves the inlined array set in memory in one shot. This procedure will
       * erase any contents that previously existed on the file. If you want to
       * append use add() instead.
       */
      void set(const InlinedArraysetImpl& set);

      /**
       * Some informative methods
       */
      inline Torch::core::array::ElementType getElementType() const { return m_elementtype; }
      inline size_t getNDim() const { return m_ndim; }
      inline const size_t* getShape() const { return m_shape; }
      inline size_t getNSamples() const { return m_samples; }

    private: //some helpers

      /**
       * Checks that the current Arrayset is compatible with the given Array.
       */ 
      void checkCompatibility (const Torch::database::Array& a) const;

      void reloadSpecification();
     
    private: //representation
      std::string m_filename; ///< The file where this array is stored
      boost::shared_ptr<const Torch::database::ArraysetCodec> m_codec; ///< How to load and save the data
      Torch::core::array::ElementType m_elementtype;
      size_t m_ndim;
      size_t m_shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
      size_t m_samples;
  };

}}}

#endif /* TORCH_DATABASE_EXTERNALARRAYSETIMPL_H */
