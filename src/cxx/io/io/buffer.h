/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  5 Oct 08:47:12 2011 
 *
 * @brief The buffer API describes a non-specific way to handle data.
 */

#ifndef TORCH_IO_BUFFER_H 
#define TORCH_IO_BUFFER_H

#include <blitz/array.h>
#include <boost/shared_ptr.hpp>
#include "core/array_type.h"

namespace Torch { namespace io {

  /**
   * Encapsulation of special type information of buffers.
   */
  struct typeinfo {

    Torch::core::array::ElementType dtype; ///< data type
    size_t nd; ///< number of dimensions
    size_t shape[TORCH_MAX_DIM]; ///< length along each dimension
    size_t stride[TORCH_MAX_DIM]; ///< strides along each dimension

    /**
     * Default constructor
     */
    typeinfo();

    /**
     * Simplification to build a typeinfo from a shape pointer.
     */
    typeinfo(Torch::core::array::ElementType dtype,
        size_t nd, const size_t* shape);

    /**
     * Copies information from another typeinfo
     */
    typeinfo(const typeinfo& other);

    /**
     * Assignment
     */
    typeinfo& operator= (const typeinfo& other);

    /**
     * Set to specific values
     */
    void set(Torch::core::array::ElementType dtype,
        size_t nd, const size_t* shape);

    /**
     * Reset to defaults -- as if uninitialized.
     */
    void reset();

    /**
     * Is this a valid type information?
     */
    bool is_valid();

    /**
     * sets the shape
     */
    void set_shape(size_t nd, const size_t* shape);

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
     * Returns the total size (in bytes) of the buffer being pointed by me.
     */
    size_t buffer_size() const;

    /**
     * Checks compatibility with other typeinfo
     */
    bool is_compatible(const typeinfo& other) const;

    /**
     * Make it easy to set for blitz::Array<T,N>
     */ 
    template <typename T, int N> void set(const blitz::Array<T,N>& array) {
      dtype = Torch::core::array::getElementType<T>();
      set_shape(array.shape());
    }

    template <typename T, int N> 
      void set(boost::shared_ptr<blitz::Array<T,N> >& array) {
        dtype = Torch::core::array::getElementType<T>();
        set_shape(array->shape());
      }

    template <int N> void set_shape(const blitz::TinyVector<int,N>& tv_shape) {
      nd = N;
      for (size_t k=0; k<nd; ++k) shape[k] = tv_shape(k);
      update_strides();
    }

  };

  /**
   * The buffer manager introduces a concept for managing the buffers that
   * can be handled as C-style arrays. It encapsulates methods to store and
   * delete the buffer contents in a safe way.
   *
   * The buffer is an entity that either stores a copy of its own data or
   * refers to data belonging to another buffer.
   */
  struct buffer {

    /**
     * By default, the buffer is never freed. You must override this method
     * to do something special for your class type.
     */
    virtual ~buffer() { }

    /**
     * Copies the data from another buffer.
     */
    virtual void set(const buffer& other) =0;

    /**
     * Refers to the data of another buffer.
     */
    virtual void set(boost::shared_ptr<buffer> other) =0;

    /**
     * Re-allocates this buffer taking into consideration new requirements. The
     * internal memory should be considered uninitialized.
     */
    virtual void set (const typeinfo& req) =0;

    /**
     * Element type
     */
    virtual const typeinfo& type() const =0;

    /**
     * Borrows a reference from the underlying memory. This means this object
     * continues to be responsible for deleting the memory and you should
     * make sure that it outlives the usage of the returned pointer.
     */
    virtual void* ptr() =0;
    virtual const void* ptr() const =0;

  };

}}

#endif /* TORCH_IO_BUFFER_H */
