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
#include "core/dataset_common.h"

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

        /**
         * This method returns a constant reference to my internal blitz array.
         * It is the fastest way to get access to my data because it involves
         * no data copying. The only downside is that you need to know the
         * correct type and number of dimensions or I'll throw an exception.
         */
        template<typename T, int D> const blitz::Array<T,D>& getData() const;

        /**
         * This method returns a constant reference to my internal data (not a
         * copy) in the type you wish. It is the easiest method to use because
         * I'll never throw, no matter which type you want to receive data at.
         * Only get the number of dimensions right!
         */
        template<typename T, int D> const blitz::Array<T,D> castData() const;

        /**
         * This method will set my internal data to the value you specify.
         * Please note you can only set to a type and number of dimensions that
         * is coherent with my current settings or my parent's settings. If you
         * wish to create a new type with different number of dimensions or
         * element type, you have to create a new array and possibly associated
         * with a new parent.
         *
         * If you are setting me for the first time, you can choose the type
         * and number of dimensions that pleases you. 
         */
        template<typename T, int D> void setData(blitz::Array<T,D>& data);

      private: //some utilities to deal with transparency with the bz arrays.

        /**
         * Loads the array data from a file into memory.
         */
        void loadBlitzArray();

        /**
         * Returns a new reference to the currently pointed blitz::Array<>, 
         * transparently. The caller becomes responsible for the returned
         * pointer. This method should only be used by myself and is
         * implemented *just* for my own internal management.
         */
        void* getBlitzArray();

        /**
         * Clones the blitz::Array in bzarray transparently. The caller becomes
         * responsible fo the returned pointer. This method should only be used
         * by myself and is implemented *just* for my own internal managament.
         */
        void* cloneBlitzArray() const;

        /**
         * Delete the underlying blitz::Array. This method should only be used
         * by myself and is implemented *just* for my own internal management.
         */
        void deleteBlitzArray();

      private: //representation
        boost::weak_ptr<Arrayset> m_parent_arrayset; ///< My current parent
        size_t m_id; ///< This is my id
        bool m_is_loaded; ///< Am I already loaded?
        std::string m_filename; ///< If I'm stored in a file, put it here
        //boost::shared_ptr<Codec> m_codec; ///< How to read from file
        Torch::core::array::ElementType m_elementtype; ///< of the blitz::Array
        size_t m_ndim; ///< The number of dimensions the blitz::Array has
        void* m_bzarray; ///< A void* to a blitz::Array instantiated here.
    };

    template<typename T, int D>
    const blitz::Array<T,D>& Array::getData() const {
      if (!m_bzarray) throw NotInitialized();
      if (D != m_ndim) throw DimensionError();
      if (Torch::core::array::getElementType<T>() != m_elementtype)
        throw TypeError();
      //at this point we know both T and D are correct and that we have an
      //internal blitz::Array<> set.
      return *static_cast<const blitz::Array<T,D> >(m_bzarray);
    }

    template<typename T, int D> const blitz::Array<T,D> Array::castData() const {
      if (!m_bzarray) throw NotInitialized();
      if (D != getNDim()) throw DimensionError();
      //at this point we know D is correct and that we have an
      //internal blitz::Array<> set.
      //TODO: Ask LES how to use blitz::complex_cast<>() in this situation...
      switch (m_elementtype) {
        case Torch::core::array::t_bool: 
          return blitz::cast<T>(*static_cast<blitz::Array<bool,D>(m_bzarray));
        case Torch::core::array::t_int8: 
          return blitz::cast<T>(*static_cast<blitz::Array<int8_t,D>(m_bzarray));
        case Torch::core::array::t_int16: 
          return blitz::cast<T>(*static_cast<blitz::Array<int16_t,D>(m_bzarray));
        case Torch::core::array::t_int32: 
          return blitz::cast<T>(*static_cast<blitz::Array<int32_t,D>(m_bzarray));
        case Torch::core::array::t_int64: 
          return blitz::cast<T>(*static_cast<blitz::Array<int64_t,D>(m_bzarray));
        case Torch::core::array::t_uint8: 
          return blitz::cast<T>(*static_cast<blitz::Array<uint8_t,D>(m_bzarray));
        case Torch::core::array::t_uint16: 
          return blitz::cast<T>(*static_cast<blitz::Array<uint16_t,D>(m_bzarray));
        case Torch::core::array::t_uint32: 
          return blitz::cast<T>(*static_cast<blitz::Array<uint32_t,D>(m_bzarray));
        case Torch::core::array::t_uint64: 
          return blitz::cast<T>(*static_cast<blitz::Array<uint64_t,D>(m_bzarray));
        case Torch::core::array::t_float32: 
          return blitz::cast<T>(*static_cast<blitz::Array<float,D>(m_bzarray));
        case Torch::core::array::t_float64: 
          return blitz::cast<T>(*static_cast<blitz::Array<double,D>(m_bzarray));
        case Torch::core::array::t_float128: 
          return blitz::cast<T>(*static_cast<blitz::Array<long double,D>(m_bzarray));
        case Torch::core::array::t_complex64: 
          return blitz::cast<T>(*static_cast<blitz::Array<std::complex<float>,D>(m_bzarray));
        case Torch::core::array::t_complex128: 
          return blitz::cast<T>(*static_cast<blitz::Array<std::complex<double>,D>(m_bzarray));
        case Torch::core::array::t_complex256: 
          return blitz::cast<T>(*static_cast<blitz::Array<std::complex<long double>,D>(m_bzarray));
      }

      //if we get to this point, there is nothing much we can do...
      throw TypeError();
    }

  } //closes namespace database

} //closes namespace Torch

#endif /* TORCH_DATABASE_ARRAY_H */

