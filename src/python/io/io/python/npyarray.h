/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 27 Oct 08:00:37 2011 
 *
 * @brief A variation of io::buffer that is built from NumPy arrays
 */

#ifndef TORCH_PYTHON_IO_NPYARRAY_H 
#define TORCH_PYTHON_IO_NPYARRAY_H

#include "core/python/pycore.h"
#include "io/buffer.h"

namespace Torch { namespace python {

  /**
   * Converts a shared io::buffer to a numpy array in the most clever way
   * possible.
   *
   * The conversion will happen in a "clever" way. Firstly we try a dynamic
   * cast from io::buffer into io::npyarray. If that succeeds, we will directly
   * return the underlying numpy array attached to the buffer given buffer
   * constraints of the underlying memory ownership.
   *
   * If the dynamic cast does not succeed it means the data was generated
   * initially on the C++ side of the running code. In this case we create a
   * new numpy array based on the data pointed by the buffer and tie the
   * lifetime of the buffer to the returned array using trick described here:
   * http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
   *
   * In this case the buffer will be marked read-only to indicate you cannot
   * change its underlying representation - you would need to copy it to do so.
   */
  PyArrayObject* buffer_array (boost::shared_ptr<Torch::io::buffer> b);

  /**
   * Wraps the buffer with a read-only shallow numpy array layer.
   */
  PyArrayObject* buffer_array (const Torch::io::buffer& b);

  /**
   * A buffer wrapper for NumPy arrays. This object is required whenever your
   * functions and methods receive numeric arrays as input and need to store
   * io::buffers inside of them.
   */
  class npyarray: public Torch::io::buffer {

    public:

      /**
       * Starts by refering to the data from a numpy array.
       */
      npyarray(PyArrayObject* array);

      /**
       * Starts by refering to the data from a numpy array.
       */
      npyarray(boost::python::numeric::array array);

      /**
       * Starts by copying the data from another buffer.
       */
      npyarray(const Torch::io::buffer& other);

      /**
       * Starts by referring to the data from another buffer.
       */
      npyarray(boost::shared_ptr<Torch::io::buffer> other);

      /**
       * Starts with an uninitialized, pre-allocated numpy array.
       */
      npyarray(const Torch::io::typeinfo& info);

      /**
       * Destroyes me
       */
      virtual ~npyarray();

      /**
       * Copies the data from another buffer.
       */
      virtual void set(const Torch::io::buffer& other);

      /**
       * Refers to the data of another buffer.
       */
      virtual void set(boost::shared_ptr<Torch::io::buffer> other);

      /**
       * Re-allocates this buffer taking into consideration new requirements.
       * The internal memory should be considered uninitialized.
       */
      virtual void set (const Torch::io::typeinfo& req);

      /**
       * Element type
       */
      virtual const Torch::io::typeinfo& type() const { return m_type; }

      /**
       * Borrows a reference from the underlying memory. This means this object
       * continues to be responsible for deleting the memory and you should
       * make sure that it outlives the usage of the returned pointer.
       */
      virtual void* ptr() { return m_ptr; }
      virtual const void* ptr() const { return m_ptr; }

      virtual boost::shared_ptr<void> owner() { return m_data; }
      virtual boost::shared_ptr<const void> owner() const { return m_data; }

      /**
       * Gets a new PyArrayObject* that refers to the data if I'm pointing to
       * to or 0. You must Py_DECREF() after you are done if the returned
       * pointer is non-zero.
       */
      PyArrayObject* shallow_copy();

      /**
       * Gets a new PyArrayObject* that contains a *copy* of the data I'm
       * pointing to. This will always succeed. You must Py_DECREF() after you
       * are done.
       */
      PyArrayObject* deep_copy();

      /**
       * Gets a new PyArrayObject* that refers to the data if I'm pointing to
       * to. Always succeeds. You must Py_DECREF() after you are done.
       * 
       * For this technique to always succeed, we use the recommendation for
       * generating the numpy arrays with a special de-allocator as found here:
       * http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory
       */
      PyArrayObject* shallow_copy_force();

    private: //representation

      Torch::io::typeinfo m_type; ///< type information
      void* m_ptr; ///< pointer to the data
      bool m_is_numpy; ///< true if initiated with a NumPy array
      boost::shared_ptr<void> m_data; ///< Pointer to the data owner

  };

}}

#endif /* TORCH_PYTHON_IO_NPYARRAY_H */
