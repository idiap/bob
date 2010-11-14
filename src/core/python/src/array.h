/**
 * @file python/src/array.h
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 *
 * @brief blitz::Array<> to and from python converters for arrays
 */

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/shared_ptr.hpp>
#include <blitz/array.h>
#include <string>
#include <map>
#include <boost/format.hpp>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <stdexcept>
#include <stdint.h>
#include <sstream>

#include "core/logging.h"

namespace Torch { namespace python {

  /**
   * This map allows me to convert from the C type code (enum) into the python 
   * type code (single char) or type name (string) and vice-versa. We
   * instantiate a static variable of this type and use it throughout the code.
   */
  struct TypeMapper { 

    public:
      /**
       * Constructor
       */
      TypeMapper();

      /** 
       * Conversion from Numpy C enum to Numpy single character typecode
       */
      const std::string& enum_to_code(NPY_TYPES t) const;

      /**
       * Conversion from Numpy C enum to string description
       */
      const std::string& enum_to_name(NPY_TYPES t) const; 

      /**
       * Conversion from Numpy C enum to blitz::Array<T,N>::T typename
       */
      const std::string& enum_to_blitzT(NPY_TYPES t) const; 

      /**
       * Converts from Numpy C enum to the size of the scalar
       */
      size_t enum_to_scalar_size(NPY_TYPES t) const;

      /**
       * Converts from Numpy C enum to the base type of the scalar
       */
      char enum_to_scalar_base(NPY_TYPES t) const;

      /**
       * Conversion from C++ type to Numpy C enum
       */
      template <typename T> NPY_TYPES type_to_enum(void) const 
      { return NPY_NOTYPE; }

      /**
       * Conversion from C++ type to Numpy typecode
       */
      template <typename T> const std::string& type_to_typecode(void) const 
      { return enum_to_code(type_to_enum<T>()); }

      /**
       * Conversion from C++ type to Numpy typename
       */
      template <typename T> const std::string& type_to_typename(void) const 
      { return enum_to_name(type_to_enum<T>()); }

      /**
       * Tells if two types are equivalent in size for the current platform
       */
      bool are_equivalent(NPY_TYPES i1, NPY_TYPES i2) const;

    private:
      std::string bind(const char* base, int size) const;
      std::string bind_typename(const char* base, const char* type, int size) const;
      const std::string& get(const std::map<NPY_TYPES, std::string>& dict,
          NPY_TYPES t) const;

      template <typename T> const typename T::mapped_type& get_raise
        (const T& dict, NPY_TYPES t) const {
          typename T::const_iterator it = dict.find(t); 
          if (it == dict.end()) {
            boost::format f("value not listed in internal mapper (%s - %s)");
            f % this->get(this->m_c_to_typecode, t);
            f % this->get(this->m_c_to_typename, t);
            PyErr_SetString(PyExc_ValueError, f.str().c_str()); 
            throw boost::python::error_already_set(); 
          }
          return it->second;
        }

    private:
      std::map<NPY_TYPES, std::string> m_c_to_typecode;
      std::map<NPY_TYPES, std::string> m_c_to_typename;
      std::map<NPY_TYPES, std::string> m_c_to_blitz;
      std::map<NPY_TYPES, size_t> m_scalar_size;
      std::map<NPY_TYPES, char> m_scalar_base;
  }; 

  template <> NPY_TYPES TypeMapper::type_to_enum<bool>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<signed char>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned char>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<short>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned short>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<int>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned int>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<long>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned long>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<long long>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned long long>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<float>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<double>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<long double>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<std::complex<float> >(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<std::complex<double> >(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<std::complex<long double> >(void) const; 

  extern struct TypeMapper TYPEMAP;

  /**
   * Returns the C Numpy enum type of the given array
   */
  NPY_TYPES type(boost::python::numeric::array a);

  /**
   * Returns the expected python blitz::Array<T,N> typename for a given numeric
   * array. Please note this is mostly trivial, except for integers where these
   * may be confused:
   * int => normally size 4 (32-bits) across architectures (32 or 64 bits)
   * long => size 4 in 32-bit archs, size 8 in 64-bit archs
   * long long => size 8 across architectures (32 or 64 bits)
   */
  std::string equivalent_scalar(boost::python::numeric::array a);

  /**
   * Checks the array type and raises a TypeError if that does not conform.
   * Please read the comments for "equivalent_scalar()" above concerning
   * exceptional cases we need to treat in this method.
   */
  template <typename T> void check_type (boost::python::numeric::array a) {
    if (!TYPEMAP.are_equivalent(type(a), TYPEMAP.type_to_enum<T>())) {
      boost::format err("expected array of type \"%s\", but got \"%s\"");
      err % TYPEMAP.type_to_typename<T>() % TYPEMAP.enum_to_name(type(a));
      PyErr_SetString(PyExc_TypeError, err.str().c_str());
      boost::python::throw_error_already_set();
    }
  }

  /**
   * Returns the number of dimensions for the given array
   */
  size_t rank(boost::python::numeric::array a);
  
  /**
   * Checks if the input array has an expected rank.
   */
  void check_rank(boost::python::numeric::array a, size_t expected_rank);

  /**
   * Checks if the given object is an array.
   */
  void check_is_array(boost::python::object o);

  /**
   * Returns a clone of this array, enforcing a new type.
   */
  boost::python::numeric::array astype(boost::python::numeric::array a, const::std::string& t);

  /**
   * Converts a Numpy array to a blitz one, using a reference to the original 
   * numpy array data. The last flag "copy" will make this function perform a
   * copy instead of just pointing to the numpy array data from the blitz array.
   */
  template<typename T, int N> boost::shared_ptr<blitz::Array<T,N> >
    numpy_to_blitz (boost::python::numeric::array a, bool copy) {

      //checks rank (i.e., number of dimensions)
      check_rank(a, N);

      //checks type
      check_type<T>(a);

      const int T_size = sizeof(T);
      blitz::TinyVector<int,N> shape(0);
      blitz::TinyVector<int,N> strides(0);
      for (int i=0;i<N;++i) {
        shape[i] = PyArray_DIM(a.ptr(), i);
        strides[i] = PyArray_STRIDE(a.ptr(), i) / T_size;
      }

      if (copy) {
        return boost::shared_ptr<blitz::Array<T,N> >(new blitz::Array<T,N>((T*)PyArray_DATA(a.ptr()), shape, strides, blitz::duplicateData));
      }

      //please note this will not copy the data, just transfer the memory
      //ownership to the returned value.
      return boost::shared_ptr<blitz::Array<T,N> >(new blitz::Array<T,N>((T*)PyArray_DATA(a.ptr()), shape, strides, blitz::neverDeleteData));
    }

  /**
   * A shortcut to make the overloading easier to solve for Boost::Python.
   */
  template<typename T, int N> boost::shared_ptr<blitz::Array<T,N> >
    numpy_to_blitz_copy (boost::python::numeric::array a)
    { return numpy_to_blitz<T,N>(a, true); }

  /**
   * Converts a blitz array to a Numpy one, in a single copy. 
   */
  template<typename T, int N>
    boost::python::numeric::array blitz_to_numpy_copy(const blitz::Array<T,N>& b) {
      if (b.base(0) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "expected array in row-major format (C-style)");
        boost::python::throw_error_already_set();
      }

      int dimensions[N];
      for (size_t i=0; i<N; ++i) { dimensions[i] = b.extent(i); }
      NPY_TYPES tp = TYPEMAP.type_to_enum<T>();
      PyArrayObject* a = (PyArrayObject*)PyArray_FromDims(N, dimensions, tp);
      T* array_data = (T*)PyArray_DATA(a);
      size_t i = 0;
      for (typename blitz::Array<T,N>::const_iterator it=b.begin(); it!=b.end(); ++it, ++i) {
        array_data[i] = *it;
      }

      boost::python::object obj(boost::python::handle<>((PyObject*)a));
      return boost::python::extract<boost::python::numeric::array>(obj);
    }

  /**
   * Returns the Numpy typecode, typename or C enum for this blitz Array
   */
  template<typename T, int N> const char* to_typecode
    (const blitz::Array<T,N>& a) {
    return TYPEMAP.type_to_typecode<T>().c_str();
  }

  template<typename T, int N> const char* to_typename
    (const blitz::Array<T,N>& a) {
    return TYPEMAP.type_to_typename<T>().c_str();
  }

  template<typename T, int N> NPY_TYPES to_enum
    (const blitz::Array<T,N>& a) {
    return TYPEMAP.type_to_enum<T>();
  }

  /**
   * Prints to a string
   */
  template<typename T, int N> 
    boost::shared_ptr<std::string> __str__(const blitz::Array<T,N>& a) {
      std::ostringstream s;
      s << a;
      return boost::shared_ptr<std::string>(new std::string(s.str()));
    }

  /**
   * This template converts any input blitz::Array<T,N> type into a pythonic
   * representation.
   */
  template<typename T, int N> 
    boost::python::class_<blitz::Array<T,N>, boost::shared_ptr<blitz::Array<T,N> > >
    array_class(const char* tname) {
      boost::format class_name("%s_%d");
      class_name % tname % N;
      boost::format class_doc("Objects of this class are a pythonic representation of blitz::Array<%s,%s>. Please refer to the blitz::Array manual for more details on the array class and its usage. The help messages attached to each member function of this binding are just for quick-referencing. (N.B. Dimensions in C-arrays are zero-indexed. The first dimension is 0, the second is 1, etc. Use the helpers 'firstDim', 'secondDim', etc to help you keeping your code clear.)");
      class_doc % tname % N;

      //base class creation
      boost::python::class_<blitz::Array<T,N>, boost::shared_ptr<blitz::Array<T,N> > > retval(class_name.str().c_str(), class_doc.str().c_str(), boost::python::init<>("Initializes an empty array"));

      //initialization from another array of the same type
      retval.def(boost::python::init<blitz::Array<T,N> >((boost::python::arg("other")), "Initializes by referencing the data from another array."));

      //initialization using extents
      switch(N) { 
        case 1: 
          retval.def(boost::python::init<int>((boost::python::arg("dim0")), "Builds array with the given size"));
          break;
        case 2: 
          retval.def(boost::python::init<int, int>((boost::python::arg("dim0"), boost::python::arg("dim1")), "Builds array with the given size"));
          retval.def("transpose", &blitz::Array<T,N>::transpose, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
          retval.def("transposeSelf", &blitz::Array<T,N>::transposeSelf, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
          break;
        case 3: 
          retval.def(boost::python::init<int, int, int>((boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2")), "Builds array with the given size"));
          retval.def("transpose", &blitz::Array<T,N>::transpose, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
          retval.def("transposeSelf", &blitz::Array<T,N>::transposeSelf, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
          break;
        case 4: 
          retval.def(boost::python::init<int, int, int, int>((boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "Builds array with the given size"));
          retval.def("transpose", &blitz::Array<T,N>::transpose, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3"), boost::python::arg("dim4")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
          retval.def("transposeSelf", &blitz::Array<T,N>::transposeSelf, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3"), boost::python::arg("dim4")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
          break;
      }

      //initialization from a numpy array
      retval.def("__init__", make_constructor(numpy_to_blitz<T,N>, boost::python::default_call_policies(), (boost::python::arg("array"), boost::python::arg("copy"))), "Builds an array copying or referring to data data from a numpy array. If `copy' is set to false, the behavior of this constructor is just like calling it only with `array'.");
      retval.def("__init__", make_constructor(numpy_to_blitz_copy<T,N>, boost::python::default_call_policies(), (boost::python::arg("array"))), "Builds an array copying data from a numpy array");

      //get a copy as numpy array
      retval.def("as_ndarray", &blitz_to_numpy_copy<T, N>, (boost::python::arg("self")), "Creates a copy of this array as a NumPy Array with the same dimensions and storage type");

      //get a copy 
      retval.def("copy", &blitz::Array<T,N>::copy, "Creates an independent copy of this array");

      //some quick operations for inspecting the array
      retval.def("__str__", &__str__<T,N>, "A stringyfied representation of myself.");
      retval.def("extent", (int (blitz::Array<T,N>::*)(int) const)&blitz::Array<T,N>::extent, (boost::python::arg("self"), boost::python::arg("dimension")), "Returns the array size in one of the dimensions");
      retval.def("dimensions", &blitz::Array<T,N>::dimensions, "Total number of dimensions on this array");
      retval.def("rank", &blitz::Array<T,N>::rank, "Total number of dimensions on this array");
      retval.def("rows", &blitz::Array<T,N>::rows, "Equivalent to extent(firstDim)");
      retval.def("columns", &blitz::Array<T,N>::columns, "Equivalent to extent(secondDim)");
      retval.def("depth", &blitz::Array<T,N>::depth, "Equivalent to extent(thirdDim)");
      retval.def("size", &blitz::Array<T,N>::size, "Total number of elements in this array");

      //operations to modify the array or to return a copy of the modified array
      retval.def("reverse", &blitz::Array<T,N>::reverse, (boost::python::arg("self"), boost::python::arg("dimension")), "This method reverses the array in the specified dimension. For example, if reverse(firstDim) is invoked on a 2-dimensional array, then the ordering of rows in the array will be reversed; reverse(secondDim) would reverse the order of the columns. Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
      retval.def("reverseSelf", &blitz::Array<T,N>::reverse, (boost::python::arg("self"), boost::python::arg("dimension")), "This method reverses the array in the specified dimension. For example, if reverse(firstDim) is invoked on a 2-dimensional array, then the ordering of rows in the array will be reversed; reverse(secondDim) would reverse the order of the columns. Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");

      //some type information
      retval.def("numpy_typecode", &to_typecode<T, N>, (boost::python::arg("self")), "Describes the equivalent numpy typecode of this blitz::Array");
      retval.def("numpy_typename", &to_typename<T, N>, (boost::python::arg("self")), "Describes the equivalent numpy typename of this blitz::Array");
      retval.def("numpy_enum", &to_enum<T, N>, (boost::python::arg("self")), "Describes the equivalent numpy C enumeration of this blitz::Array");

      return retval;
    };

} } //namespace Torch::python 

#define declare_arrays(T,NAME) void BOOST_JOIN(bind_core_array_,NAME)() { \
  Torch::python::array_class<T,1>(BOOST_STRINGIZE(NAME)); \
  Torch::python::array_class<T,2>(BOOST_STRINGIZE(NAME)); \
  Torch::python::array_class<T,3>(BOOST_STRINGIZE(NAME)); \
  Torch::python::array_class<T,4>(BOOST_STRINGIZE(NAME)); \
}
