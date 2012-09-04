/**
 * @file python/core/core/python/ndarray.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief A boost::python extension object that plays the role of a NumPy
 * ndarray (PyArrayObject*) and bob::core::array::interface at the same time.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_PYTHON_NDARRAY_H
#define BOB_PYTHON_NDARRAY_H

#include <boost/python.hpp> //this has to come before the next declaration!
#include <boost/format.hpp>

// ============================================================================
// Note: Header files that are distributed and include numpy/arrayobject.h need
//       to have these protections. Be warned.

// Defines a unique symbol for the API
#if !defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PY_ARRAY_UNIQUE_SYMBOL bob_NUMPY_ARRAY_API
#endif

// Normally, don't import_array(), except if bob_IMPORT_ARRAY is defined.
#if !defined(bob_IMPORT_ARRAY) and !defined(NO_IMPORT_ARRAY)
#define NO_IMPORT_ARRAY
#endif

// Finally, we include numpy's arrayobject header. Not before!
#include <numpy/arrayobject.h>
// ============================================================================

#include "bob/core/python/exception.h"
#include "bob/core/array.h"
#include "bob/core/cast.h"
#include "bob/core/Exception.h"

#include <blitz/array.h>
#include <stdint.h>

/**
 * The method object::is_none() was only introduced in boost v1.43.
 */
#if BOOST_VERSION >= 104300
#define TPY_ISNONE(x) x.is_none()
#else
#define TPY_ISNONE(x) (x.ptr() == Py_None)
#endif

/**
 * A macro that is replaced by the proper format definition for size_t
 */
#ifdef __LP64__
#  define SIZE_T_FMT "%lu"
#else
#  define SIZE_T_FMT "%u"
#endif

namespace bob { namespace python {

  /**
   * Initializes numpy and boost bindings. Should be called once per module.
   *
   * Pass to it the module doc string and it will also update the module
   * documentation string.
   */
  void setup_python(const char* module_docstring);

  /**
   * A generic method to convert from ndarray type_num to bob's ElementType
   */
  bob::core::array::ElementType num_to_type(int num);

  /**
   * A method to retrieve the type of element of an array
   */
  bob::core::array::ElementType array_to_type(const boost::python::numeric::array& a);

  /**
   * Retrieves the number of dimensions in an array
   */
  size_t array_to_ndim(const boost::python::numeric::array& a);

  /**
   * Converts from C/C++ type to ndarray type_num.
   */
  template <typename T> int ctype_to_num(void) {
    PYTHON_ERROR(TypeError, "unsupported C/C++ type (%s)", bob::core::array::stringize<T>());
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
#ifdef NPY_FLOAT128
  template <> int ctype_to_num<long double>(void);
#endif
  template <> int ctype_to_num<std::complex<float> >(void);
  template <> int ctype_to_num<std::complex<double> >(void);
#ifdef NPY_COMPLEX256
  template <> int ctype_to_num<std::complex<long double> >(void);
#endif

  /**
   * Converts from bob's Element type to ndarray type_num
   */
  int type_to_num(bob::core::array::ElementType type);

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
      bob::core::array::typeinfo& i);

  /**
   * This is the same as above, but does not run any check on the input object
   * "o".
   */
  void typeinfo_ndarray_ (const boost::python::object& o,
      bob::core::array::typeinfo& i);

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
      bob::core::array::typeinfo& info, bool writeable=true,
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
   * 3. If 2. holds, shape values are checked if has_valid_shape() is 'true'
   */
  convert_t convertible_to (boost::python::object array_like,
      const bob::core::array::typeinfo& info, bool writeable=true,
      bool behaved=true);

  /**
   * Same as above, but only requires dtype convertibility.
   */
  convert_t convertible_to (boost::python::object array_like,
      boost::python::object dtype_like, bool writeable=true,
      bool behaved=true);

  /**
   * Same as above, but requires nothing, just simple convertibility.
   */
  convert_t convertible_to (boost::python::object array_like,
      bool writeable=true, bool behaved=true);

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
       * Builds a new dtype object from a bob element type
       */
      dtype(bob::core::array::ElementType eltype);

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
      bool has_type(bob::core::array::ElementType eltype) const; ///< matches

      /**
       * Returns the current element type
       */
      bob::core::array::ElementType eltype() const;

      /**
       * Returns the current type num or -1, if I'm None
       */
      int type_num() const;

      /**
       * Returns a boost::python representation of this object - maybe None.
       */
      inline boost::python::object self() const { return m_self; }

      /**
       * Returns the bp::str() object for myself
       */
      boost::python::str str() const;

      /**
       * Returns str(*this) as a std::string
       */
      std::string cxx_str() const;

    private: //representation

      boost::python::object m_self;

  };

  class py_array: public bob::core::array::interface {

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
      py_array(boost::python::object array_like,
              boost::python::object dtype_like);

      /**
       * Builds a new array copying the data of an existing buffer.
       */
      py_array(const bob::core::array::interface& buffer);

      /**
       * Builds a new array by referring to the data of an existing buffer.
       */
      py_array(boost::shared_ptr<bob::core::array::interface> buffer);

      /**
       * Builds a new array from scratch using the typeinfo. This array will be
       * a NumPy ndarray internally.
       */
      py_array(const bob::core::array::typeinfo& info);

      template <typename T>
      py_array(bob::core::array::ElementType t, T d0) {
        set(bob::core::array::typeinfo(t, (T)1, &d0));
      }
      template <typename T>
      py_array(bob::core::array::ElementType t, T d0, T d1) {
        T shape[2] = {d0, d1};
        set(bob::core::array::typeinfo(t, (T)2, &shape[0]));
      }
      template <typename T>
      py_array(bob::core::array::ElementType t, T d0, T d1, T d2) {
        T shape[3] = {d0, d1, d2};
        set(bob::core::array::typeinfo(t, (T)3, &shape[0]));
      }
      template <typename T>
      py_array(bob::core::array::ElementType t, T d0, T d1, T d2, T d3) {
        T shape[4] = {d0, d1, d2, d3};
        set(bob::core::array::typeinfo(t, (T)4, &shape[0]));
      }
      template <typename T>
      py_array(bob::core::array::ElementType t, T d0, T d1, T d2, T d3, T d4)
      {
        T shape[5] = {d0, d1, d2, d3, d4};
        set(bob::core::array::typeinfo(t, (T)5, &shape[0]));
      }

      /**
       * D'tor virtualization
       */
      virtual ~py_array();

      /**
       * Copies the data from another buffer.
       */
      virtual void set(const bob::core::array::interface& buffer);

      /**
       * Refers to the data of another buffer.
       */
      virtual void set(boost::shared_ptr<bob::core::array::interface> buffer);

      /**
       * Re-allocates this buffer taking into consideration new requirements.
       * The internal memory should be considered uninitialized.
       */
      virtual void set (const bob::core::array::typeinfo& req);

      /**
       * Type information for this buffer.
       */
      virtual const bob::core::array::typeinfo& type() const { return m_type; }

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
      virtual boost::python::object copy
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
      virtual boost::python::object pyobject();

      /**
       * Tells if the buffer is writeable
       */
      virtual bool is_writeable() const; ///< PyArray_ISWRITEABLE

    private: //representation

      bob::core::array::typeinfo m_type; ///< type information
      void* m_ptr; ///< pointer to the data
      bool m_is_numpy; ///< true if initiated with a NumPy array
      boost::shared_ptr<void> m_data; ///< Pointer to the data owner

  };

  /**
   * The ndarray class is just a smart pointer wrapper over the concrete
   * implementation of py_array.
   */
  class ndarray {

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
       * Builds a new array from an array-like object but coerces to a certain
       * type.
       *
       * @param array_like An ndarray object, inherited type or any object that
       * can be cast into an array. Note that, in case of casting, we will need
       * to copy the data. Otherwise, we just refer.
       */
      ndarray(boost::python::object array_like);

      /**
       * Builds a new array from scratch using a type and shape
       */
      ndarray(const bob::core::array::typeinfo& info);

      template <typename T>
      ndarray(bob::core::array::ElementType t, T d0)
        : px(new py_array(t, d0)) { }
      template <typename T>
      ndarray(bob::core::array::ElementType t, T d0, T d1)
        : px(new py_array(t, d0, d1)) { }
      template <typename T>
      ndarray(bob::core::array::ElementType t, T d0, T d1, T d2)
        : px(new py_array(t, d0, d1, d2)) { }
      template <typename T>
      ndarray(bob::core::array::ElementType t, T d0, T d1, T d2, T d3)
        : px(new py_array(t, d0, d1, d2, d3)) { }
      template <typename T>
      ndarray(bob::core::array::ElementType t, T d0, T d1, T d2, T d3, T d4)
        : px(new py_array(t, d0, d1, d2, d3, d4)) { }

      /**
       * D'tor virtualization
       */
      virtual ~ndarray();

      /**
       * Returns the type information
       */
      virtual const bob::core::array::typeinfo& type() const;

      /**
       * Returns the underlying python representation.
       */
      virtual boost::python::object self();

      /**
       * Returns a temporary blitz::Array<> skin over this ndarray.
       *
       * Attention: If you use this method, you have to make sure that this
       * ndarray outlives the blitz::Array<> and that such blitz::Array<> will
       * not be re-allocated or have any other changes made to it, except for
       * the data contents.
       */
      template <typename T, int N> blitz::Array<T,N> bz () {

        typedef blitz::Array<T,N> array_type;
        typedef blitz::TinyVector<int,N> shape_type;

        const bob::core::array::typeinfo& info = px->type();

        if (info.nd != N) {
          boost::format mesg("cannot wrap numpy.ndarray(%s,%d) as blitz::Array<%s,%s> - dimensions do not match");
          mesg % bob::core::array::stringize(info.dtype) % info.nd;
          mesg % bob::core::array::stringize<T>() % N;
          throw std::invalid_argument(mesg.str().c_str());
        }

        if (info.dtype != bob::core::array::getElementType<T>()) {
          boost::format mesg("cannot wrap numpy.ndarray(%s,%d) as blitz::Array<%s,%s> - data type does not match");
          mesg % bob::core::array::stringize(info.dtype) % info.nd;
          mesg % bob::core::array::stringize<T>() % N;
          throw std::invalid_argument(mesg.str().c_str());
        }

        shape_type shape;
        shape_type stride;
        for (size_t k=0; k<info.nd; ++k) {
          shape[k] = info.shape[k];
          stride[k] = info.stride[k];
        }

        //finally, we return the wrapper.
        return array_type((T*)px->ptr(), shape, stride, blitz::neverDeleteData);
      }

    protected: //representation

      boost::shared_ptr<py_array> px;

  };

  /**
   * A specialization of ndarray that is used to cast types from python that
   * will **not** be modified in C++.
   *
   * Conversion requirements for this type can be made less restrictive since
   * we consider the user just wants to pass a value to the method or function
   * using this type. This opposes to the plain ndarray, in which the user may
   * want to modify its contents by skinning it with a blitz::Array<> layer.
   */
  class const_ndarray: public ndarray {

    public: //api

      /**
       * Builds a new array from an array-like object but coerces to a certain
       * type.
       *
       * @param array_like An ndarray object, inherited type or any object that
       * can be cast into an array. Note that, in case of casting, we will need
       * to copy the data. Otherwise, we just refer.
       */
      const_ndarray(boost::python::object array_like);

      /**
       * D'tor virtualization
       */
      virtual ~const_ndarray();

      /**
       * Returns a temporary blitz::Array<> skin over this const_ndarray, if possible,
       * otherwise it will COPY the array to the requested type and returns the copy.
       *
       * Attention: If you use this method, you have to make sure that this
       * ndarray outlives the blitz::Array<>, in case the data is not copied.
       */
      template <typename T, int N> const blitz::Array<T,N> cast() {
        const bob::core::array::typeinfo& info = px->type();

        if (info.nd != N) {
          boost::format mesg("cannot wrap numpy.ndarray(%s,%d) as blitz::Array<%s,%s> - dimensions do not match");
          mesg % bob::core::array::stringize(info.dtype) % info.nd;
          mesg % bob::core::array::stringize<T>() % N;
          throw std::invalid_argument(mesg.str().c_str());
        }

        if (info.dtype == bob::core::array::getElementType<T>()) {
          // Type and shape matches, return the shallow copy of the array.
          return bz<T,N>();
        }

        // if we got here, we have to copy-cast
        // call the correct version of the cast function
        switch(info.dtype){
          // boolean types
          case bob::core::array::t_bool: return bob::core::cast<T>(bz<bool,N>());

          // integral types
          case bob::core::array::t_int8: return bob::core::cast<T>(bz<int8_t,N>());
          case bob::core::array::t_int16: return bob::core::cast<T>(bz<int16_t,N>());
          case bob::core::array::t_int32: return bob::core::cast<T>(bz<int32_t,N>());
          case bob::core::array::t_int64: return bob::core::cast<T>(bz<int64_t,N>());

          // unsigned integral types
          case bob::core::array::t_uint8: return bob::core::cast<T>(bz<uint8_t,N>());
          case bob::core::array::t_uint16: return bob::core::cast<T>(bz<uint16_t,N>());
          case bob::core::array::t_uint32: return bob::core::cast<T>(bz<uint32_t,N>());
          case bob::core::array::t_uint64: return bob::core::cast<T>(bz<uint64_t,N>());

          // floating point types
          case bob::core::array::t_float32: return bob::core::cast<T>(bz<float,N>());
          case bob::core::array::t_float64: return bob::core::cast<T>(bz<double,N>());
          case bob::core::array::t_float128: return bob::core::cast<T>(bz<long double,N>());

          // complex types
          case bob::core::array::t_complex64: return bob::core::cast<T>(bz<std::complex<float>,N>());
          case bob::core::array::t_complex128: return bob::core::cast<T>(bz<std::complex<double>,N>());
          case bob::core::array::t_complex256: return bob::core::cast<T>(bz<std::complex<long double>,N>());
          
          default: throw bob::core::NotImplementedError();
        }
      }

  };

}}

#endif /* BOB_PYTHON_NDARRAY_H */
