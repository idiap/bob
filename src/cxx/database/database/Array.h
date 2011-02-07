/**
 * @file database/src/Array.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * The Array is the basic unit containing data in a Dataset 
 */

#ifndef TORCH_DATABASE_ARRAY_H 
#define TORCH_DATABASE_ARRAY_H

#include <cstdlib>
#include <stdint.h>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <blitz/array.h>

#include "database/InlinedArrayImpl.h"
#include "database/ExternalArrayImpl.h"

namespace Torch {   
  /**
   * \ingroup libdatabase_api
   * @{
   *
   */
  namespace database {

    //I promise this exists:
    class Arrayset;

    /**
     * The array class for a dataset
     */
    class Array {

      public:

        /**
         * This initializes a new Array with a default id that serves as a
         * placeholder. As soon as this array is adopted by some Arrayset, the
         * id will be correctly set to the next available id in that set
         */
        Array();

        /**
         * Builds an Array that contains data from a file. You can optionally
         * specify the name of a codec and the array id.
         */
        Array(const std::string& filename="", const std::string& codec="");

        /**
         * Copies the Array data from another array. If any blitz::Array<>
         * is allocated internally, its data will be <b>copied</b> by this new 
         * array as well. If a parent was already set, the id property of this
         * copy will be reset according to the availability of ids in the
         * parent.
         */
        Array(const Array& other);

        /**
         * Destroys this array. If this array had a parent, this array is
         * firstly unregistered from the parent and then destroyed together
         * with its data.
         */
        virtual ~Array();

        /**
         * Copies the data from another array. If any blitz::Array<> is
         * allocated internally, its data will be <b>copied</b> by this array as
         * well. If a parent was already set, the id property of this copy will
         * be reset according to the availability of ids in the parent.
         */
        Array& operator= (const Array& other);

        /**
         * Is the current array empty?
         */
        inline bool isEmpty() const { return m_bzarray; }

        /**
         * Make the current array empty, not pointing or containing any data.
         * Please note that the parent and id settings remain unchanged, if
         * they were already set.
         */
        void clear();

        /**
         * Set the filename containing the data. An empty string
         * indicates that the data are stored inlined at the database. If codec
         * is empty it means "figure it out by using the filename extension".
         *
         * The codec will be looked up and the file poked for the element type
         * and the number of dimensions that it can provide.
         */
        void setFilename(const std::string& filename, 
            const std::string& codecname="");

        /**
         * Get the filename containing the data if any. An empty string
         * indicates that the data is stored inlined.
         */
        inline const std::string& getFilename() const { return m_filename; }

        /**
         * TODO:
         * Get the codec used to read the data from the external file 
         * if any. This will be non-empty if the filename is non-empty.
         */
        //inline boost::shared_ptr<Codec> getCodec() const { return m_codec; }

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
         * Sets the parent arrayset of this array. If my parent is already set,
         * I'll detach myself from my old parent and insert myself onto the new
         * parent, checking initialized type information if any was already
         * set. Incompatibilities will be flagged by exceptions.
         *
         * The optional parameter 'id' will set my id property, but first I'll
         * check if that id is unblocked in the parent. If that is not the
         * case, I'll raise an exception. If you don't set the id property in
         * this call, I'll ask the parent to assign me a proper available id.
         */
        void setParent (boost::shared_ptr<Arrayset> parent, size_t id=0);

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

      private: //representation
        boost::weak_ptr<Arrayset> m_parent_arrayset; ///< My current parent
        size_t m_id; ///< This is my id
    };

  } //closes namespace database

} //closes namespace Torch

#endif /* TORCH_DATABASE_ARRAY_H */

