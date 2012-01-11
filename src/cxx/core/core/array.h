/**
 * @file cxx/core/core/array.h
 * @date Tue Nov 8 15:34:31 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief The array API describes a non-specific way to handle N dimensional
 * array data.
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BOB_CORE_ARRAY_INTERFACE_H 
#define BOB_CORE_ARRAY_INTERFACE_H

#include <stdexcept>
#include <string>

#include <boost/shared_ptr.hpp>
#include <blitz/array.h>

#include "core/array_type.h"

namespace bob { namespace core { namespace array {

  /**
   * Encapsulation of special type information of interfaces.
   */
  struct typeinfo {

    ElementType dtype; ///< data type
    size_t nd; ///< number of dimensions
    size_t shape[BOB_MAX_DIM+1]; ///< length along each dimension
    size_t stride[BOB_MAX_DIM+1]; ///< strides along each dimension

    /**
     * Default constructor
     */
    typeinfo();

    /**
     * Simplification to build a typeinfo from a size
     */
    template <typename T> typeinfo(ElementType dtype_, T nd_) {
      set(dtype_, nd_);
    }

    /**
     * Simplification to build a typeinfo from a shape pointer.
     */
    template <typename T> typeinfo(ElementType dtype_, T nd_, const T* shape_) {
      set(dtype_, nd_, shape_);
    }

    /**
     * Copies information from another typeinfo
     */
    typeinfo(const typeinfo& other);

    /**
     * Assignment
     */
    typeinfo& operator= (const typeinfo& other);

    /**
     * Builds with type and number of dimensions, but set the shape and strides
     * to all zeros.
     */
    template <typename T>
    void set(ElementType dtype_, T nd_) {
      dtype = dtype_;
      nd = nd_;
      reset_shape();
    }

    /**
     * Set to specific values
     */
    template <typename T>
    void set(ElementType dtype_, T nd_, const T* shape_) {
      dtype = dtype_;
      set_shape(nd_, shape_);
    }

    /**
     * Set to specific values, including strides
     */
    template <typename T>
    void set(ElementType dtype_, T nd_, const T* shape_,
        const T* stride_) {
      dtype = dtype_;
      nd = nd_;
      for (size_t k=0; k<nd; ++k) {
        shape[k] = shape_[k];
        stride[k] = stride_[k];
      }
    }

    /**
     * Reset to defaults -- as if uninitialized.
     */
    void reset();

    /**
     * Is this a valid type information?
     */
    bool is_valid() const;

    /**
     * Does this has a valid shape information?
     */
    bool has_valid_shape() const;

    /**
     * sets the shape
     */
    template <typename T> void set_shape(T nd_, const T* shape_) {
      if (nd_ > (BOB_MAX_DIM+1))
        throw std::invalid_argument("unsupported number of dimensions");
      nd = nd_;
      for (size_t k=0; k<nd; ++k) shape[k] = shape_[k];
      update_strides();
    }

    /**
     * resets the shape to all zeros
     */
    void reset_shape();

    /**
     * Update my own stride vector. Called automatically after any use of
     * set_shape().
     */
    void update_strides();

    /**
     * Returns the total number of elements available
     */
    size_t size() const;

    /**
     * Returns the size of each element
     */
    inline size_t item_size() const { return getElementSize(dtype); }

    /**
     * Returns the total size (in bytes) of the buffer that I'm associated
     * with.
     */
    size_t buffer_size() const;

    /**
     * Returns the item type description
     */
    const char* item_str() const { return stringize(dtype); }

    /**
     * Checks compatibility with other typeinfo
     */
    bool is_compatible(const typeinfo& other) const;

    /**
     * Formats and returns a string containing the full typeinfo description.
     */
    std::string str() const;

    /**
     * Make it easy to set for blitz::Array<T,N>
     */ 
    template <typename T, int N> void set(const blitz::Array<T,N>& array) {
      dtype = getElementType<T>();
      set_shape(array.shape());
    }

    template <typename T, int N> 
      void set(boost::shared_ptr<blitz::Array<T,N> >& array) {
        dtype = getElementType<T>();
        set_shape(array->shape());
      }

    template <int N> void set_shape(const blitz::TinyVector<int,N>& tv_shape) {
      nd = N;
      for (size_t k=0; k<nd; ++k) shape[k] = tv_shape(k);
      update_strides();
    }

  };

  /**
   * The interface manager introduces a concept for managing the interfaces that
   * can be handled as C-style arrays. It encapsulates methods to store and
   * delete the buffer contents in a safe way.
   *
   * The interface is an entity that either stores a copy of its own data or
   * refers to data belonging to another interface.
   */
  class interface {

    public: //api

      /**
       * By default, the interface is never freed. You must override this method
       * to do something special for your class type.
       */
      virtual ~interface() { }

      /**
       * Copies the data from another interface.
       */
      virtual void set(const interface& other) =0;

      /**
       * Refers to the data of another interface.
       */
      virtual void set(boost::shared_ptr<interface> other) =0;

      /**
       * Re-allocates this interface taking into consideration new requirements.
       * The internal memory should be considered uninitialized.
       */
      virtual void set (const typeinfo& req) =0;

      /**
       * Type information for this interface.
       */
      virtual const typeinfo& type() const =0;

      /**
       * Borrows a reference from the underlying memory. This means this object
       * continues to be responsible for deleting the memory and you should
       * make sure that it outlives the usage of the returned pointer.
       */
      virtual void* ptr() =0;
      virtual const void* ptr() const =0;

      /**
       * Returns a representation of the internal cache using shared pointers.
       */
      virtual boost::shared_ptr<void> owner() =0;
      virtual boost::shared_ptr<const void> owner() const =0;

  };

}}}

#endif /* BOB_CORE_ARRAY_INTERFACE_H */
