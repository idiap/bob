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
   * Creates an auto-deletable bp::object out of a standard Python object that
   * cannot be NULL. Can be Py_NONE.
   *
   * Effects:
   *
   * The PyObject* is **not** XINCREF'ed at construction.
   * The PyObject* is XDECREF'ed at destruction.
   */
  boost::python::object make_non_null_object(PyObject* obj);

  /**
   * Creates an auto-deletable bp::object out of a standard Python object, that
   * may be NULL (or Py_NONE).
   *
   * Effects:
   *
   * The PyObject* is **not** XINCREF'ed at construction.
   * The PyObject* is XDECREF'ed at destruction.
   */
  boost::python::object make_maybe_null_object(PyObject* obj);

  /**
   * Creates an auto-deletable bp::object out of a standard Python object. The
   * input object cannot be NULL, but can be Py_NONE.
   *
   * Effects:
   *
   * The PyObject* is XINCREF'ed at construction.
   * The PyObject* is XDECREF'ed at destruction.
   */
  boost::python::object make_non_null_borrowed_object(PyObject* obj);

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

  /**
   * Handles conversion checking possibilities
   */
  typedef enum {
    IMPOSSIBLE = 0,    ///< not possible to get array from object
    BYREFERENCE = 1,   ///< possible, by only referencing the array
    WITHARRAYCOPY = 2, ///< possible, object is an array, but has to copy
    WITHCOPY = 3       ///< possible, object is not an array, has to copy
  } convert_t;

  /**
   * Extracts the typeinfo object from a numeric::array (passed as
   * boost::python::object). We check the input object to assure it is a valid
   * ndarray. An exception may be thrown otherwise.
   */
  void typeinfo_ndarray (const boost::python::object& o, 
      Torch::core::array::typeinfo& i);

  /**
   * This is the same as above, but does not run any check on the input object
   * "o".
   */
  void typeinfo_ndarray_ (const boost::python::object& o, 
      Torch::core::array::typeinfo& i);

  /**
   * Checks if an array-like object is convertible to become a NumPy ndarray
   * (boost::python::numeric::array). If so, write the typeinfo information
   * that such array would have upon automatic conversion to "info". 
   *
   * Optionally, you can specify you do *not* want writeable or behavior to be
   * checked. Write-ability means that an array area can be extracted from the
   * "array_like" object and changes done to the converted ndarray will be
   * reflected upon the original object.
   *
   * Behavior refers to two settings: first, the data type byte-order should be
   * native (i.e., little-endian on little-endian machines and big-endian on
   * big-endian machines). Secondly, the array must be C-Style, have its memory
   * aligned and on a contiguous block.
   *
   * This method is more efficient than actually performing the conversion,
   * unless you compile the project against NumPy < 1.6 in which case the
   * built-in checks are not available and you we will emulate them with
   * brute-force conversion if required. A level-1 DEBUG message will be output
   * if a brute-force copy is required so you can debug for that.
   *
   * This method returns the convertibility status for the array-like object,
   * which is one of:
   *
   * * IMPOSSIBLE: The object cannot, possibly, be converted into an ndarray
   * * BYREFERENCE: The object will successfuly be converted to a ndarray, i.e.
   *                in the most optimal way - by referring to it.
   * * WITHARRAYCOPY: The object will successfuly be converted to a ndarray,
   *                  but that will require an array copy. That means the
   *                  object is already an array, but not of the type you
   *                  requested.
   * * WITHCOPY: The object will successfuly be converted to a ndarray, but
   *             we will need to convert the object from its current format
   *             (non-ndarray) to a ndarray format. In this case, we will not
   *             be able to implement write-back.
   */
  convert_t convertible(boost::python::object array_like, 
      Torch::core::array::typeinfo& info, bool writeable=true,
      bool behaved=true);

  /**
   * This method does the same as convertible(), but specifies a type
   * information to which the destination array needs to have. Same rules
   * apply.
   *
   * The typeinfo input is honoured like this:
   *
   * 1. The "dtype" component is enforced on the array object
   * 2. If "nd" != 0, the number of dimensions is checked.
   * 3. If 2. holds, shape values that are **different** than zero are checked.
   */
  convert_t convertible_to (boost::python::object array_like,
      const Torch::core::array::typeinfo& info, bool writeable=true,
      bool behaved=true);

  /**
   * Same as above, but only requires dtype convertibility.
   */
  convert_t convertible_to (boost::python::object array_like, 
      boost::python::object dtype_like, bool writeable=true, 
      bool behaved=true);

  class dtype {

    public: //api

      /**
       * Builds a new dtype object from another object.
       */
      dtype (boost::python::object dtype_like);

      /**
       * Builds a new dtype object from a PyArray_Descr object that will have
       * its own reference counting increased internally. So, the object is
       * *not* stolen and you can Py_(X)DECREF() it when done if you so wish.
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

