/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  1 Nov 10:15:43 2011 
 *
 * @brief A boost::python extension object that plays the role of a NumPy
 * ndarray (PyArrayObject*) and Torch::core::array::interface at the same time.
 */

#ifndef TORCH_PYTHON_NDARRAY_H 
#define TORCH_PYTHON_NDARRAY_H

#include <boost/python.hpp> //this has to come before the next declaration!

// ============================================================================
// Note: Header files that are distributed and include numpy/arrayobject.h need
//       to have these protections. Be warned.

// Defines a unique symbol for the API
#if !defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PY_ARRAY_UNIQUE_SYMBOL torch_NUMPY_ARRAY_API
#endif

// Normally, don't import_array(), except if torch_IMPORT_ARRAY is defined.
#if !defined(torch_IMPORT_ARRAY) and !defined(NO_IMPORT_ARRAY)
#define NO_IMPORT_ARRAY
#endif

// Finally, we include numpy's arrayobject header. Not before!
#include <numpy/arrayobject.h>
// ============================================================================

#include "core/python/exception.h"
#include "core/array.h"

namespace Torch { namespace python {

  /**
   * Initializes numpy and boost bindings. Should be called once per module.
   *
   * Pass to it the module doc string and it will also update the module
   * documentation string.
   */
  void setup_python(const char* module_docstring);

  /**
   * A generic method to convert from ndarray type_num to torch's ElementType
   */
  Torch::core::array::ElementType num_to_type(int num);

  /**
   * Converts from C/C++ type to ndarray type_num.
   */
  template <typename T> int ctype_to_num(void) {
    PYTHON_ERROR(TypeError, "unsupported C/C++ type");
  }

  // The C/C++ types we support should be declared here.
  template <> int ctype_to_num<bool>(void); 
  template <> int ctype_to_num<signed char>(void); 
  template <> int ctype_to_num<unsigned char>(void); 
  template <> int ctype_to_num<short>(void); 
  template <> int ctype_to_num<unsigned short>(void); 
  template <> int ctype_to_num<int>(void); 
  template <> int ctype_to_num<unsigned int>(void); 
  template <> int ctype_to_num<long>(void);
  template <> int ctype_to_num<unsigned long>(void);
  template <> int ctype_to_num<long long>(void);
  template <> int ctype_to_num<unsigned long long>(void);
  template <> int ctype_to_num<float>(void);
  template <> int ctype_to_num<double>(void); 
  template <> int ctype_to_num<long double>(void); 
  template <> int ctype_to_num<std::complex<float> >(void);
  template <> int ctype_to_num<std::complex<double> >(void); 
  template <> int ctype_to_num<std::complex<long double> >(void); 

  /**
   * Converts from torch's Element type to ndarray type_num
   */
  int type_to_num(Torch::core::array::ElementType type);

  class dtype {

    public: //api

      /**
       * Builds a new dtype object from another object.
       */
      dtype (boost::python::object dtype_like);

      /**
       * Builds a new dtype object from a PyArray_Descr object that will have
       * its own reference counting increased internally. So, the object is
       * *not* borrowed/stolen and you can delete it when done if you so wish.
       */
      dtype (PyArray_Descr* descr);

      /**
       * Builds a new dtype object from a numpy type_num integer
       */
      dtype(int npy_typenum);

      /**
       * Builds a new dtype object from a torch element type
       */
      dtype(Torch::core::array::ElementType eltype);

      /**
       * Copy constructor
       */
      dtype(const dtype& other);

      /**
       * Default constructor -- use default dtype from NumPy
       */
      dtype();

      /**
       * D'tor virtualization
       */ 
      virtual ~dtype();

      /**
       * Assignment
       */
      dtype& operator= (const dtype& other);

      /**
       * Somme checks
       */
      bool has_native_byteorder() const; ///< byte order is native
      bool has_type(Torch::core::array::ElementType eltype) const; ///< matches

      /**
       * Returns the current element type
       */
      Torch::core::array::ElementType eltype() const;

      /**
       * Returns the current type num or -1, if I'm None
       */
      int type_num() const;

      /**
       * Returns a boost::python representation of this object - maybe None.
       */
      inline boost::python::object self() const { return m_self; }

    private: //representation

      boost::python::object m_self;

  };

  class ndarray: public Torch::core::array::interface {

    public: //api

      /**
       * Builds a new array from an array-like object but coerces to a certain
       * type.
       *
       * @param array_like An ndarray object, inherited type or any object that
       * can be cast into an array. Note that, in case of casting, we will need
       * to copy the data. Otherwise, we just refer.
       *
       * @param dtype_like Anything that can be cast to a description type.
       */
      ndarray(boost::python::object array_like,
              boost::python::object dtype_like);

      /**
       * Builds a new array copying the data of an existing buffer.
       */
      ndarray(const Torch::core::array::interface& buffer);

      /**
       * Builds a new array by referring to the data of an existing buffer.
       */
      ndarray(boost::shared_ptr<Torch::core::array::interface> buffer);

      /**
       * Builds a new array from scratch using the typeinfo. This array will be
       * a NumPy ndarray internally.
       */
      ndarray(const Torch::core::array::typeinfo& info);

      /**
       * D'tor virtualization
       */
      virtual ~ndarray();

      /**
       * Copies the data from another buffer.
       */
      virtual void set(const Torch::core::array::interface& buffer);

      /**
       * Refers to the data of another buffer.
       */
      virtual void set(boost::shared_ptr<Torch::core::array::interface> buffer);

      /**
       * Re-allocates this buffer taking into consideration new requirements.
       * The internal memory should be considered uninitialized.
       */
      virtual void set (const Torch::core::array::typeinfo& req);

      /**
       * Type information for this buffer.
       */
      virtual const Torch::core::array::typeinfo& type() const { return m_type; }

      /**
       * Borrows a reference from the underlying memory. This means this object
       * continues to be responsible for deleting the memory and you should
       * make sure that it outlives the usage of the returned pointer.
       */
      virtual void* ptr() { return m_ptr; }
      virtual const void* ptr() const { return m_ptr; }

      /**
       * Gets a handle to the owner of this buffer.
       */
      virtual boost::shared_ptr<void> owner() { return m_data; }
      virtual boost::shared_ptr<const void> owner() const { return m_data; }

      /**
       * Cast the array to a different type by copying. If the type is omitted,
       * we just make a plain copy of this array.
       */
      boost::python::object copy
        (const boost::python::object& dtype = boost::python::object());

      /**
       * Gets a shallow copy of this array, if internally it is a NumPy array.
       * Otherwise, returns a wrapper around the internal buffer memory and
       * correctly reference counts it so the given object becomes responsible
       * for the internal buffer as well.
       * 
       * For this technique to always succeed, we use the recommendation for
       * generating the numpy arrays with a special de-allocator as found here:
       * http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory
       */
      boost::python::object pyobject();

      /**
       * Tells if the buffer is writeable
       */
      bool is_writeable() const; ///< PyArray_ISWRITEABLE

    private: //representation

      Torch::core::array::typeinfo m_type; ///< type information
      void* m_ptr; ///< pointer to the data
      bool m_is_numpy; ///< true if initiated with a NumPy array
      boost::shared_ptr<void> m_data; ///< Pointer to the data owner

  };

}}

#endif /* TORCH_PYTHON_NDARRAY_H */

