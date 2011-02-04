/**
 * @file database/src/Array.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * The Array is the basic unit containing data in a Dataset 
 */

#ifndef TORCH_DATABASE_ARRAY_H 
#define TORCH_DATABASE_ARRAY_H

#include <cstdlib>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <blitz/array.h>
#include <stdint.h>

#include "core/array_common.h"

namespace Torch {   
  /**
   * \ingroup libdatabase_api
   * @{
   *
   */
  namespace database {

    //I promise this exists:
    //class Codec;
    class Arrayset;

    /**
     * The array class for a dataset
     */
    class Array {

      public:

        /**
         * This initializes a new and a default id that serves as a
         * placeholder. As soon as this array is adopted by some Arrayset, the
         * id will be correctly set to the next available id in that set. You
         * can also specify the id in this constructor.
         */
        Array(size_t id=0);

        /**
         * Builds an Array that contains data from a file. You can optionally
         * specify the name of a codec and the array id.
         */
        Array(const std::string& filename="", const std::string& codec="",
            size_it id=0);

        /**
         * Copies the Array data from another array. If any blitz::Array<>
         * is allocated internally, its data will not be copied, but just
         * referred by the blitz::Array in this new Array object. 
         */
        Array(const Array& other);

        /**
         * Destructor virtualization
         */
        virtual ~Array();

        /**
         * Copies the data from another array. If any blitz::Array<> is
         * allocated internally, its data will not be copied, but just referred
         * by the blitz::Array in this new Array object.
         */
        Array& operator= (const Array& other);

        /**
         * Is the current array empty?
         */
        inline bool isEmpty() const { return m_bzarray; }

        /**
         * Make the current array empty, not pointing or containing any data.
         * Please note that the parent and id settings remain unchanged.
         */
        void clear();

        /**
         * Set the filename containing the data if any. An empty string
         * indicates that the data are stored inlined at the database. If codec
         * is empty it means "figure it out by using the filename extension"
         */
        void setFilename(const std::string& filename,
            const std::string& codecname="");

        /**
         * Get the filename containing the data if any. An empty string
         * indicates that the data is stored inlined.
         */
        inline const std::string& getFilename() const { return m_filename; }

        /**
         * Get the codec used to read the data from the external file 
         * if any. This will be non-empty if the filename is non-empty.
         */
        //inline boost::shared_ptr<Codec> getCodec() const { return m_codec; }

        /**
         * Sets the id of the Array
         */
        inline void setId(size_t id) const { m_id = id; }

        /**
         * Gets the id of the Array
         */
        inline size_t getId() const { return m_id; }

        /**
         * Get the flag indicating if the array is loaded from an 
         * external file.
         */
        inline bool isLoaded() const { return m_is_loaded; }

        /**
         * Sets the parent arrayset of this array
         */
        inline void setParent (boost::shared_ptr<Arrayset> parent)
        { m_parent(parent); }

        /**
         * Gets the parent arrayset of this array
         */
        inline boost::shared_ptr<const Arrayset> getParent() const 
        { return m_parent_arrayset.lock(); }
        
        inline boost::shared_ptr<Arrayset> getParent()
        { return m_parent_arrayset.lock(); }

        /**
         * Returns some information about this array
         */
        inline size_t getNDim() const { return m_ndim; }
        inline Torch::core::array::ElementType getElementType() const 
        { return m_elementtype; }

        /**
         * Adapts the size of each dimension of the passed blitz array
         * to the ones of the underlying array and copy the data in it.
         */
        template<typename T, int D> blitz::Array<T,D> data() const;

        /**
         * Adapts the size of each dimension of the passed blitz array
         * to the ones of the underlying array and refer to the data in it.
         * @warning Updating the content of the blitz array will update the
         * content of the corresponding array in the dataset.
         */
        template<typename T, int D> blitz::Array<T,D> data();

      private: //some utilities to deal with transparency with the bz arrays.

        /**
         * Returns a reference to the currently pointed blitz::Array<>, 
         * transparently. This method should only be used by myself.
         */
        void* getBlitzArray() const;

        /**
         * Clones the blitz::Array in bzarray transparently. This method should
         * only be used by myself.
         */
        void* cloneBlitzArray() const;

        /**
         * Delete the underlying blitz::Array.
         */
        void deleteBlitzArray() const;

        /**
         * Returns a version of the underlying array cast into the type desired
         * by the user. Please note that the number of dimensions (i.e. the
         * array rank) must match or an exception will be thrown.
         */
        blitz::Array<T,D> getBlitzArrayWithCast() const;

      private:
        /**
         * Copies the data from the Array object into the given C-style
         * array, and cast if required.
         */
        template <typename T, typename U> void copyCast( U* out) const;

        /**
         * Checks that the number of dimensions match in order to be
         * able to refer to the data.
         */
        template <int D> void referCheck() const;

      private: //representation
        boost::weak_ptr<Arrayset> m_parent_arrayset; ///< My current
        size_t m_id; ///< This is my id
        bool m_is_loaded; ///< Am I already loaded?
        std::string m_filename; ///< If I'm stored in a file, put it here
        //boost::shared_ptr<Codec> m_codec; ///< How to read from file
        Torch::core::array::ElementType m_elementtype; ///< of the blitz::Array
        size_t m_ndim; ///< The number of dimensions the blitz::Array has
        void* m_bzarray; ///< A void* to a blitz::Array instantiated here.
    };

  } //closes namespace database

} //closes namespace Torch

#endif /* TORCH_DATABASE_ARRAY_H */

