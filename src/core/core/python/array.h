/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief blitz::Array<> to and from python converters for arrays
 */

#ifndef TORCH_CORE_PYTHON_ARRAY
#define TORCH_CORE_PYTHON_ARRAY

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>
#include <blitz/array.h>
#include <string>
#include <stdint.h>
#include <sstream>

#include "core/logging.h"
#include "core/python/ndarray.h"
#include "core/python/TypeMapper.h"

//some blitz extras for signed and unsigned 8 and 16-bit integers
namespace blitz {
  // abs(int8_t)
  template<>
  struct Fn_abs< int8_t > {
    typedef int8_t T_numtype1;
    typedef int8_t T_numtype;

    static T_numtype
      apply(T_numtype1 a) { return BZ_MATHFN_SCOPE(abs)(a); }

    template<typename T1>
      static void prettyPrint(BZ_STD_SCOPE(string) &str,
          prettyPrintFormat& format, const T1& t1) {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
      }
  };

  // abs(int16_t)
  template<> struct Fn_abs< int16_t > {
    typedef int16_t T_numtype1;
    typedef int16_t T_numtype;

    static T_numtype
      apply(T_numtype1 a) { return BZ_MATHFN_SCOPE(abs)(a); }

    template<typename T1>
      static void prettyPrint(BZ_STD_SCOPE(string) &str,
          prettyPrintFormat& format, const T1& t1) {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
      }
  };

#if !defined(__LP64__) || defined(__APPLE__)
  // abs(int64_t)
  template<> struct Fn_abs< int64_t > {
    typedef int64_t T_numtype1;
    typedef int64_t T_numtype;

    static T_numtype
      apply(T_numtype1 a) { return BZ_MATHFN_SCOPE(abs)(a); }

    template<typename T1>
      static void prettyPrint(BZ_STD_SCOPE(string) &str,
          prettyPrintFormat& format, const T1& t1) {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
      }
  };
  // missing scalar ops
  BZ_DECLARE_ARRAY_ET_SCALAR_OPS(int64_t)
  BZ_DECLARE_ARRAY_ET_SCALAR_OPS(uint64_t)
#endif

}

namespace Torch { namespace python {

  template <typename T, int N>
  boost::python::ndarray make_ndarray(const blitz::Array<T,N>& bzarray) {
    boost::python::ndarray npy(bzarray);
    return npy;
  }

  template <typename T, int N>
  boost::shared_ptr<blitz::Array<T,N> > make_blitz(const boost::python::ndarray& array) {
    return array.to_blitz<T,N>();
  }

  /**
   * This methods implement the __getitem__ functionality expected by every
   * python random access container. Here is their behavior:
   * 1. Checks if the received index is a valid entry w.r.t. N and the size of
   * the input array. Here are the possibilities
   *    a) If index is a single integer, N has to be 1, the index has to be
   *    smaller than the first array extent. The same is applicable if index is
   *    a single slice element.
   *    b) If index is a tuple composed of integers, N == len(index) and every
   *    index has to be smaller than the extent they refer to. The same is
   *    applicable if the tuple is composed of a mix of integers and slices.
   * 2. If the input index refers to a single element of the array, we return
   * this value as a python object. If the input index refers to a slice, we
   * return a new array referencing the array in the positions selected.
   *
   * Negative indexing is supported, mimmicking normal python random access
   * containers.
   */

  /**
   * We specialize to avoid computing N at run time.
   *
   * With N == 1 the input index can be either a single number, an iterable
   * with a single number, a range or an iterable with a single range
   */

  /**
   * This method will check the array limits in one dimension and will raise an
   * error if appropriate. It returns the C++ index expected for the
   * blitz::Array<> operation in case of no problems. Fortran-style arrays are
   * also taken into account.
   */
  int check_array_limits(int index, int base, int extent);

  //gets from tuple (remember TinyVector == tuple!)
  template <typename T, int N> T __getitem__(const blitz::Array<T,N>& a,
      const blitz::TinyVector<int,N>& index) {
    blitz::TinyVector<int,N> use;
    for (int i=0; i<N; ++i)
      use[i] = check_array_limits(index[i], a.base(i), a.extent(i)); 
    return a(use);
  }
  template <typename T, int N> void __setitem__(blitz::Array<T,N>& a,
      const blitz::TinyVector<int,N>& index, const T& v) {
    blitz::TinyVector<int,N> use;
    for (int i=0; i<N; ++i) 
      use[i] = check_array_limits(index[i], a.base(i), a.extent(i)); 
    a(use) = v;
  }

  //gets from a integer (1-D special case)
  template<typename T> inline T __getitem_1__ (blitz::Array<T,1>& a,
      int index) { 
    return a(check_array_limits(index, a.base(0), a.extent(0))); 
  }
  template <typename T> inline void __setitem_1__ (blitz::Array<T,1>& a,
      int index, const T& v) {
    a(check_array_limits(index, a.base(0), a.extent(0))) = v; 
  }

  //gets a slice from

  /**
   * Constructs a new Torch/Blitz array starting from a python sequence. The
   * data from the python object is copied.
   */
  template<typename T, int N> boost::shared_ptr<blitz::Array<T,N> >
    iterable_to_blitz (boost::python::object o, const blitz::TinyVector<int,N>& shape, blitz::GeneralArrayStorage<N> storage) {
      /**
       * Conditions we have to check for:
       * 1. The object "o" has to be iterable
       * 2. The number of elements in o has to be exactly the total defined by
       * the shape.
       * 3. All elements in "o" have to be convertible to T
       */
      bool type_check = PyList_Check(o.ptr()) || PyTuple_Check(o.ptr()) ||
        PyIter_Check(o.ptr());
      if (!type_check) {
        PyErr_SetString(PyExc_TypeError, "input object has to be of type list, tuple or iterable");
        boost::python::throw_error_already_set();
      }
      /**
       * Check length (item 2)
       */
      Py_ssize_t length = PyObject_Length(o.ptr());
      if (length != blitz::product(shape)) {
        boost::format s("input object does not contain %d elements, but %d");
        s % blitz::product(shape) % length;
        PyErr_SetString(PyExc_RuntimeError, s.str().c_str());
        boost::python::throw_error_already_set();
      }
      /**
       * This bit will run the filling and will check at the same time
       */
      boost::shared_ptr<blitz::Array<T,N> >retval(new blitz::Array<T,N>(shape, storage));
      typename blitz::Array<T,N>::iterator j(retval->begin());
      boost::python::handle<> obj_iter(PyObject_GetIter(o.ptr()));
      for(Py_ssize_t i=0; i<length;++i,++j) {
        boost::python::handle<> py_elem_hdl(
            boost::python::allow_null(PyIter_Next(obj_iter.get())));
        if (PyErr_Occurred()) {
          PyErr_Clear();
          boost::format s("element %d is not accessible?");
          s % i;
          PyErr_SetString(PyExc_RuntimeError, s.str().c_str());
          boost::python::throw_error_already_set();
        }
        if (!py_elem_hdl.get()) break; // end of iteration
        boost::python::object py_elem_obj(py_elem_hdl);
        (*j) = boost::python::extract<T>(py_elem_obj);
      }

      return retval;
    }
  
  template<typename T, int N> boost::shared_ptr<blitz::Array<T,N> >
    iterable_to_blitz_c (boost::python::object o, const blitz::TinyVector<int,N>& shape) {
      return iterable_to_blitz<T,N>(o, shape, blitz::GeneralArrayStorage<N>());
    }

  /**
   * Returns the Numpy typecode, typename or C enum for this blitz Array
   */
  template<typename T, int N> const char* to_typecode
    (const blitz::Array<T,N>& a) {
    return Torch::python::TYPEMAP.type_to_typecode<T>().c_str();
  }

  template<typename T, int N> const char* to_typename
    (const blitz::Array<T,N>& a) {
    return Torch::python::TYPEMAP.type_to_typename<T>().c_str();
  }

  template<typename T, int N> NPY_TYPES to_enum
    (const blitz::Array<T,N>& a) {
    return Torch::python::TYPEMAP.type_to_enum<T>();
  }

  /**
   * Prints to a string
   */
  template<typename T, int N> 
    boost::python::str __str__(const blitz::Array<T,N>& a) {
      std::ostringstream s;
      s << a;
      return boost::python::str(s.str());
    }

  /**
   * Casts to a certain type
   */
  template <typename T, int N, typename T2> blitz::Array<T2,N> cast(blitz::Array<T,N>& i) { return blitz::Array<T2,N>(blitz::cast<T2>(i)); }

  /**
   * Some arithmetic operators
   */
  template <typename T, int N> blitz::Array<T,N> __add__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<T,N>(i1 + i2); }
  template <typename T, int N> blitz::Array<T,N> __add_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<T,N>(i1 + i2); }
  template <typename T, int N> blitz::Array<T,N> __sub__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<T,N>(i1 - i2); }
  template <typename T, int N> blitz::Array<T,N> __sub_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<T,N>(i1 - i2); }
  template <typename T, int N> blitz::Array<T,N> __mul__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<T,N>(i1 * i2); }
  template <typename T, int N> blitz::Array<T,N> __mul_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<T,N>(i1 * i2); }
  template <typename T, int N> blitz::Array<T,N> __div__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<T,N>(i1 / i2); }
  template <typename T, int N> blitz::Array<T,N> __div_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<T,N>(i1 / i2); }
  template <typename T, int N> blitz::Array<T,N> __mod__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<T,N>(i1 % i2); }
  template <typename T, int N> blitz::Array<T,N> __mod_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<T,N>(i1 % i2); }
 
  template <typename T, int N> blitz::Array<bool,N> __lt__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<bool,N>(i1 < i2); }
  template <typename T, int N> blitz::Array<bool,N> __lt_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<bool,N>(i1 < i2); }
  template <typename T, int N> blitz::Array<bool,N> __le__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<bool,N>(i1 <= i2); }
  template <typename T, int N> blitz::Array<bool,N> __le_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<bool,N>(i1 <= i2); }
  template <typename T, int N> blitz::Array<bool,N> __gt__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<bool,N>(i1 > i2); }
  template <typename T, int N> blitz::Array<bool,N> __gt_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<bool,N>(i1 > i2); }
  template <typename T, int N> blitz::Array<bool,N> __ge__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<bool,N>(i1 >= i2); }
  template <typename T, int N> blitz::Array<bool,N> __ge_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<bool,N>(i1 >= i2); }
  template <typename T, int N> blitz::Array<bool,N> __ne__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<bool,N>(i1 != i2); }
  template <typename T, int N> blitz::Array<bool,N> __ne_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<bool,N>(i1 != i2); }
  template <typename T, int N> blitz::Array<bool,N> __eq__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<bool,N>(i1 == i2); }
  template <typename T, int N> blitz::Array<bool,N> __eq_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<bool,N>(i1 == i2); }
  template <typename T, int N> blitz::Array<bool,N> __and__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<bool,N>(i1 & i2); }
  template <typename T, int N> blitz::Array<bool,N> __and_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<bool,N>(i1 & i2); }
  template <typename T, int N> blitz::Array<bool,N> __or__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<bool,N>(i1 | i2); }
  template <typename T, int N> blitz::Array<bool,N> __or_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<bool,N>(i1 | i2); }
  template <typename T, int N> blitz::Array<bool,N> __xor__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<bool,N>(i1 ^ i2); }
  template <typename T, int N> blitz::Array<bool,N> __xor_c__(blitz::Array<T,N>& i1, const T& i2) { return blitz::Array<bool,N>(i1 ^ i2); }

  template <typename T, int N> blitz::Array<T,N> __invert__(blitz::Array<T,N>& i) { return blitz::Array<T,N>(~i); }
  template <typename T, int N> blitz::Array<T,N> __neg__(blitz::Array<T,N>& i) { return blitz::Array<T,N>(-i); }
  template <typename T, int N> blitz::Array<T,N> __abs__(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::abs(i)); }
  
  template <typename T, int N> blitz::Array<T,N> acos(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::acos(i)); }
  template <typename T, int N> blitz::Array<T,N> asin(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::asin(i)); }
  template <typename T, int N> blitz::Array<T,N> atan(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::atan(i)); }
  template <typename T, int N> blitz::Array<T,N> cos(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::cos(i)); }
  template <typename T, int N> blitz::Array<T,N> cosh(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::cosh(i)); }
  template <typename T, int N> blitz::Array<T,N> acosh(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::acosh(i)); }
  template <typename T, int N> blitz::Array<T,N> log(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::log(i)); }
  template <typename T, int N> blitz::Array<T,N> log10(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::log10(i)); }
  template <typename T, int N> blitz::Array<T,N> sin(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::sin(i)); }
  template <typename T, int N> blitz::Array<T,N> sinh(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::sinh(i)); }
  template <typename T, int N> blitz::Array<T,N> sqr(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::sqr(i)); }
  template <typename T, int N> blitz::Array<T,N> sqrt(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::sqrt(i)); }
  template <typename T, int N> blitz::Array<T,N> tan(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::tan(i)); }
  template <typename T, int N> blitz::Array<T,N> tanh(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::tanh(i)); }
  template <typename T, int N> blitz::Array<T,N> atanh(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::atanh(i)); }
  template <typename T, int N> blitz::Array<T,N> cbrt(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::cbrt(i)); }
  template <typename T, int N> blitz::Array<T,N> exp(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::exp(i)); }
  template <typename T, int N> blitz::Array<T,N> expm1(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::expm1(i)); }
  template <typename T, int N> blitz::Array<T,N> erf(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::erf(i)); }
  template <typename T, int N> blitz::Array<T,N> erfc(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::erfc(i)); }
  template <typename T, int N> blitz::Array<T,N> ilogb(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::ilogb(i)); }
  //TODO: template <typename T, int N> blitz::Array<int,N> isnan(blitz::Array<T,N& i) { return blitz::Array<int,N>(blitz::blitz_isnan(i)); }
  template <typename T, int N> blitz::Array<T,N> j0(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::j0(i)); }
  template <typename T, int N> blitz::Array<T,N> j1(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::j1(i)); }
  template <typename T, int N> blitz::Array<T,N> lgamma(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::lgamma(i)); }
  template <typename T, int N> blitz::Array<T,N> log1p(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::log1p(i)); }
  template <typename T, int N> blitz::Array<T,N> rint(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::rint(i)); }
  template <typename T, int N> blitz::Array<T,N> y0(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::y0(i)); }
  template <typename T, int N> blitz::Array<T,N> y1(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::y1(i)); }

  //operate on floats
  template <typename T, int N> blitz::Array<T,N> ceil(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::ceil(i)); }
  template <typename T, int N> blitz::Array<T,N> floor(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::floor(i)); }

  //operate on complex T
  template <typename T, int N> blitz::Array<T,N> arg(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::arg(i)); }
  template <typename T, int N> blitz::Array<T,N> conj(blitz::Array<T,N>& i) { return blitz::Array<T,N>(blitz::conj(i)); }

  //some reductions
  //TODO: Missing reductions that take a dimension parameter (please note this
  //is not an "int". Blitz provides its own scheme with indexes which are fully
  //fledged types. See the manual.
  template <typename T, int N> T sum(blitz::Array<T,N>& i) { return blitz::sum(i); }
  //template <typename T, int N> blitz::Array<T,1> sum_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::sum(i, dim)); }
  template <typename T, int N> T product(blitz::Array<T,N>& i) { return blitz::product(i); }
  //template <typename T, int N> blitz::Array<T,1> product_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::product(i, dim)); }
  template <typename T, int N> T mean(blitz::Array<T,N>& i) { return blitz::mean(i); }
  //template <typename T, int N> blitz::Array<T,1> mean_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::mean(i, dim)); }
  template <typename T, int N> T min(blitz::Array<T,N>& i) { return blitz::min(i); }
  //template <typename T, int N> blitz::Array<T,1> min_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::min(i, dim)); }
  template <typename T, int N> T max(blitz::Array<T,N>& i) { return blitz::max(i); }
  //template <typename T, int N> blitz::Array<T,1> max_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::max(i, dim)); }
  template <typename T, int N> blitz::TinyVector<int,N> minIndex(blitz::Array<T,N>& i) { return blitz::minIndex(i); }
  //template <typename T, int N> blitz::TinyVector<int,N> minIndex_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::minIndex(i, dim)); }
  template <typename T, int N> blitz::TinyVector<int,N> maxIndex(blitz::Array<T,N>& i) { return blitz::maxIndex(i); }
  //template <typename T, int N> blitz::TinyVector<int,N> maxIndex_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::maxIndex(i, dim)); }
  template <typename T, int N> int count(blitz::Array<T,N>& i) { return blitz::count(i); }
  //template <typename T, int N> blitz::Array<int,1> count_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::count(i, dim)); }
  template <typename T, int N> bool any(blitz::Array<T,N>& i) { return blitz::any(i); }
  //template <typename T, int N> blitz::Array<bool,1> any_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::any(i, dim)); }
  template <typename T, int N> bool all(blitz::Array<T,N>& i) { return blitz::all(i); }
  //template <typename T, int N> blitz::Array<bool,1> all_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::all(i, dim)); }
  
  //Some special functions with 2 arguments
  template <typename T, int N> blitz::Array<T,N> __pow__(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) { return blitz::Array<T,N>(blitz::pow(i1, i2)); }
  template <typename T, int N> blitz::Array<T,N> __pow_c__(blitz::Array<T,N>& i1, const T& i2) { blitz::Array<T,N> tmp(i1.shape()); tmp = i2; return blitz::Array<T,N>(blitz::pow(i1, tmp)); }

  //Functions to help filling
  template <typename T, int N> void fill(blitz::Array<T,N>& i, const T& v) { i = v; }
  template <typename T, int N> void zeroes(blitz::Array<T,N>& i) { i = 0; }
  template <typename T, int N> void ones(blitz::Array<T,N>& i) { i = 1; }

  /**
   * This struct will declare the several bits that are part of the Torch
   * blitz::Array<T,N> python bindings.
   */
  template<typename T, int N> struct array_binder {

    public:
      //declares a few typedefs to make it easier to write code
      typedef typename blitz::Array<T,N> array_type;
      typedef typename blitz::TinyVector<int,N> shape_type;
      typedef typename blitz::GeneralArrayStorage<N> storage_type;
      typedef array_type& (array_type::*inplace_const_op)(const T&); 
      typedef array_type& (array_type::*inplace_array_op)(const array_type&);
      typedef boost::python::class_<array_type, boost::shared_ptr<array_type> >
        bp_array_type;
      
      //constants that can help you a bit
      static std::string s_element_type_str;
      static std::string s_blitz_type_str;

    private:
      boost::shared_ptr<bp_array_type> m_class;


    public:
      /**
       * The constructor does the basics for the initialization of objects of
       * this class. It will call all boost methods to build the bindings.
       */
      array_binder(const char* tname) {
        s_element_type_str = tname;
        boost::format blitz_name("blitz::Array<%s,%d>");
        blitz_name % tname % N;
        s_blitz_type_str = blitz_name.str();
        boost::format class_name("%s_%d");
        class_name % tname % N;
        boost::format class_doc("Objects of this class are a pythonic representation of blitz::Array<%s,%s>. Please refer to the blitz::Array manual for more details on the array class and its usage. The help messages attached to each member function of this binding are just for quick-referencing. (N.B. Dimensions in C-arrays are zero-indexed. The first dimension is 0, the second is 1, etc. Use the helpers 'firstDim', 'secondDim', etc to help you keeping your code clear.)");
        class_doc % tname % N;

        //base class creation
        m_class = boost::shared_ptr<bp_array_type>(new bp_array_type(class_name.str().c_str(), class_doc.str().c_str(), boost::python::init<>("Initializes an empty array")));
        m_class->def_readonly("cxx_element_typename", &s_element_type_str); 
        m_class->def_readonly("cxx_blitz_typename", &s_blitz_type_str); 

        load_init();
        load_indexers();
        load_io();
        load_copiers();
        load_resizers();
        load_informers();
        load_basic_arith();
        load_transformers();
        load_math_operators();
      }

      /**
       * Binds initializers
       */
      void load_init() {
        //intialization with only the storage type
        m_class->def(boost::python::init<storage_type>(boost::python::arg("storage"), "Constructs a new array with a specific storage, but with no contents"));

        //initialization from another array of the same type
        m_class->def(boost::python::init<array_type>((boost::python::arg("other")), "Initializes by referencing the data from another array."));

        //initialization using extents
        switch(N) { 
          case 1: 
            m_class->def(boost::python::init<int>((boost::python::arg("dim0")), "Builds array with the given size"));
            m_class->def(boost::python::init<int,storage_type>((boost::python::arg("dim0"), boost::python::arg("storage")), "Builds array with the given size and a storage type."));
            break;
          case 2: 
            m_class->def(boost::python::init<int,int>((boost::python::arg("dim0"), boost::python::arg("dim1")), "Builds array with the given size"));
            m_class->def(boost::python::init<int,int,storage_type>((boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("storage")), "Builds array with the given size and a storage type."));
            break;
          case 3: 
            m_class->def(boost::python::init<int, int, int>((boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2")), "Builds array with the given size"));
            m_class->def(boost::python::init<int,int,int,storage_type>((boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("storage")), "Builds array with the given size and a storage type."));
            break;
          case 4: 
            m_class->def(boost::python::init<int, int, int, int>((boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "Builds array with the given size"));
            m_class->def(boost::python::init<int,int,int,int,storage_type>((boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3"), boost::python::arg("storage")), "Builds array with the given size and a storage type."));
            break;
        }

        //initialization using a TinyVector<int,T> (bound to tuple)
        m_class->def(boost::python::init<const shape_type&>((boost::python::arg("extent")), "Initalizes the array with extents described in a tuple"));
        m_class->def(boost::python::init<const shape_type&,storage_type>((boost::python::arg("extent"), boost::python::arg("storage")), "Initalizes the array with extents described in a tuple and a storage type."));

        //initialization from a numpy array or iterable
        m_class->def("__init__", make_constructor(iterable_to_blitz<T,N>, boost::python::default_call_policies(), (boost::python::arg("iterable"), boost::python::arg("shape"), boost::python::arg("storage"))), "Builds an array from a python sequence. Please note that the length of the sequence or iterable must be exactly the same as defined by the array shape parameter. You should also specify a storage order (GeneralArrayStorage or FortranArray).");
        m_class->def("__init__", make_constructor(iterable_to_blitz_c<T,N>, boost::python::default_call_policies(), (boost::python::arg("iterable"), boost::python::arg("shape"))), "Builds an array from a python sequence. Please note that the length of the sequence or iterable must be exactly the same as defined by the array shape parameter. This version will build a C-storage.");
        m_class->def("__init__", make_constructor(&make_blitz<T,N>, boost::python::default_call_policies(), (boost::python::arg("array"))), "Builds an array copying data from a numpy array");

      }

      /**
       * TODO: Loads indexers and slicers
       */
      void load_indexers() {
        m_class->def("__getitem__", &__getitem__<T,N>, (boost::python::arg("self"), boost::python::arg("tuple")), "Accesses one element of the array."); 
        m_class->def("__setitem__", &__setitem__<T,N>, (boost::python::arg("self"), boost::python::arg("tuple"), boost::python::arg("value")), "Sets one element of the array."); 
        if (N == 1) { //loads special case for 1-D arrays
          m_class->def("__getitem__", &__getitem_1__<T>, (boost::python::arg("self"), boost::python::arg("index")), "Accesses one element of the array."); 
          m_class->def("__setitem__", &__setitem_1__<T>, (boost::python::arg("self"), boost::python::arg("index"), boost::python::arg("value")), "Sets one element of the array."); 
        }
      }

      /**
       * Loads stuff related to I/O
       */
      void load_io() {
        m_class->def("__str__", &__str__<T,N>, "A stringyfied representation of myself.");
        //TODO: input
      }

      /**
       * Loads stuff for exporting the array
       */
      void load_copiers() {
        //get a copy as numpy array
        m_class->def("as_ndarray", &make_ndarray<T,N>, (boost::python::arg("self")), "Creates a copy of this array as a NumPy Array with the same dimensions and storage type");
        m_class->def("copy", &array_type::copy, "This method creates a copy of the array's data, using the same storage ordering as the current array. The returned array is guaranteed to be stored contiguously in memory, and to be the only object referring to its memory block (i.e. the data isn't shared with any other array object).");
        m_class->def("makeUnique", &array_type::makeUnique, "If the array's data is being shared with another Blitz++ array object, this member function creates a copy so the array object has a unique view of the data.");
      }

      /**
       * Loads operators for resizing
       */
      void load_resizers() {
        m_class->def("free", &array_type::free, "This method resizes an array to zero size. If the array data is not being shared with another array object, then it is freed.");
        m_class->def("resize", (void (array_type::*)(const shape_type&))(&array_type::resize), boost::python::arg("shape"), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
        m_class->def("resizeAndPreserve", (void (array_type::*)(const shape_type&))(&array_type::resizeAndPreserve), boost::python::arg("shape"), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");

        switch(N) { 
          case 1: 
            m_class->def("resize", (void (array_type::*)(int))(&array_type::resize), (boost::python::arg("self"), boost::python::arg("dim0")), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
            //bogus blitz implementation:
            //m_class->def("resizeAndPreserve", (void (array_type::*)(int))(&array_type::resizeAndPreserve), (boost::python::arg("self"), boost::python::arg("dim0")), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");
            break;
          case 2: 
            m_class->def("resize", (void (array_type::*)(int,int))(&array_type::resize), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1")), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
            //bogus blitz implementation:
            //m_class->def("resizeAndPreserve", (void (array_type::*)(int,int))(&array_type::resizeAndPreserve), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1")), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");
            break;
          case 3: 
            m_class->def("resize", (void (array_type::*)(int,int,int))(&array_type::resize), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2")), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
            //bogus blitz implementation:
            //m_class->def("resizeAndPreserve", (void (array_type::*)(int,int,int))(&array_type::resizeAndPreserve), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2")), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");
            break;
          case 4: 
            m_class->def("resize", (void (array_type::*)(int,int,int,int))(&array_type::resize), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
            //bogus blitz implementation:
            //m_class->def("resizeAndPreserve", (void (array_type::*)(int,int,int,int))(&array_type::resizeAndPreserve), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");
            break;
        }

      }

      /**
       * Loads all kind of informative methods
       */
      void load_informers() {
        //some quick operations for inspecting the array
        m_class->def("extent", (int (array_type::*)(int) const)&array_type::extent, (boost::python::arg("self"), boost::python::arg("dimension")), "Returns the array size in one of the dimensions");
        m_class->def("dimensions", &array_type::dimensions, "Total number of dimensions on this array");
        m_class->def("rank", &array_type::rank, "Total number of dimensions on this array");
        m_class->def("rows", &array_type::rows, "Equivalent to extent(firstDim)");
        m_class->def("columns", &array_type::columns, "Equivalent to extent(secondDim)");
        m_class->def("depth", &array_type::depth, "Equivalent to extent(thirdDim)");
        m_class->def("size", &array_type::size, "Total number of elements in this array");
        m_class->def("base", (const shape_type& (array_type::*)() const)(&array_type::base), boost::python::return_value_policy<boost::python::return_by_value>(), "The base of a dimension is the first valid index value. A typical C-style array will have base of zero; a Fortran-style array will have base of one. The base can be different for each dimension, but only if you deliberately use a Range-argument constructor or design a custom storage ordering.");
        m_class->def("base", (int (array_type::*)(int) const)(&array_type::base), "The base of a dimension is the first valid index value. A typical C-style array will have base of zero; a Fortran-style array will have base of one. The base can be different for each dimension, but only if you deliberately use a Range-argument constructor or design a custom storage ordering.");
        m_class->def("isMajorRank", &array_type::isMajorRank, boost::python::arg("dimension"), "Returns true if the dimension has the largest stride. For C-style arrays, the first dimension always has the largest stride. For Fortran-style arrays, the last dimension has the largest stride.");
        m_class->def("isMinorRank", &array_type::isMinorRank, boost::python::arg("dimension"), "Returns true if the dimension does not have the largest stride. See also isMajorRank().");
        m_class->def("isStorageContiguous", &array_type::isStorageContiguous, "Returns true if the array data is stored contiguously in memory. If you slice the array or work on subarrays, there can be skips -- the array data is interspersed with other data not part of the array. See also the various data..() functions. If you need to ensure that the storage is contiguous, try reference(copy()).");
        m_class->def("numElements", &array_type::numElements, "The same as size()");
        m_class->def("shape", &array_type::shape, boost::python::return_value_policy<boost::python::return_by_value>(), "Returns the vector of extents (lengths) of the array");
        m_class->def("stride", (const shape_type& (array_type::*)() const)&array_type::stride, boost::python::return_value_policy<boost::python::return_by_value>(), "A stride is the distance between pointers to two array elements which are adjacent in a dimension. For example, A.stride(firstDim) is equal to &A(1,0,0) - &A(0,0,0). The stride for the second dimension, A.stride(secondDim), is equal to &A(0,1,0) - &A(0,0,0), and so on. For more information about strides, see the description of custom storage formats in Section 2.9 of the Blitz manual. See also the description of parameters like firstDim and secondDim in the previous section of the same manual.");
        m_class->def("stride", (int (array_type::*)(int) const)&array_type::stride, "A stride is the distance between pointers to two array elements which are adjacent in a dimension. For example, A.stride(firstDim) is equal to &A(1,0,0) - &A(0,0,0). The stride for the second dimension, A.stride(secondDim), is equal to &A(0,1,0) - &A(0,0,0), and so on. For more information about strides, see the description of custom storage formats in Section 2.9 of the Blitz manual. See also the description of parameters like firstDim and secondDim in the previous section of the same manual.");
      
        //some type information that correlates the C++ type to the Numpy C-API
        //types.
        m_class->def("numpy_typecode", &to_typecode<T, N>, (boost::python::arg("self")), "Describes the equivalent numpy typecode of this blitz::Array");
        m_class->def("numpy_typename", &to_typename<T, N>, (boost::python::arg("self")), "Describes the equivalent numpy typename of this blitz::Array");
        m_class->def("numpy_enum", &to_enum<T, N>, (boost::python::arg("self")), "Describes the equivalent numpy C enumeration of this blitz::Array");

      }

      /**
       * Loads all math arithmetic operators that are applicable to all types
       */
      void load_basic_arith() {
        typedef array_type& (array_type::*inplace_const_op)(const T&); 
        typedef array_type& (array_type::*inplace_array_op)(const array_type&);
        m_class->def("__add__", &__add__<T,N>, "Adds two arrays element-wise"); 
        m_class->def("__add__", &__add_c__<T,N>, "Adds an array with a constant element-wise"); 
        m_class->def("__sub__", &__sub__<T,N>, "Subtracts two arrays element-wise"); 
        m_class->def("__sub__", &__sub_c__<T,N>, "Subtracts an array with a constant element-wise"); 
        m_class->def("__mul__", &__mul__<T,N>, "Multiplies two arrays element-wise");
        m_class->def("__mul__", &__mul_c__<T,N>, "Multiplies an array with a constant element-wise"); 
        m_class->def("__div__", &__div__<T,N>, "Divides two arrays element-wise"); 
        m_class->def("__div__", &__div_c__<T,N>, "Divides an array with a constant element-wise"); 
        m_class->def("__iadd__", (inplace_const_op)&array_type::operator+=, boost::python::return_self<>(), "Inplace addition with constant.");
        m_class->def("__iadd__", (inplace_array_op)&array_type::operator+=, boost::python::return_self<>(), "Inplace addition with array, elementwise.");
        m_class->def("__isub__", (inplace_const_op)&array_type::operator-=, boost::python::return_self<>(), "Inplace subtraction by constant.");
        m_class->def("__isub__", (inplace_array_op)&array_type::operator-=, boost::python::return_self<>(), "Inplace subtraction by array, elementwise.");
        m_class->def("__imul__", (inplace_const_op)&array_type::operator*=, boost::python::return_self<>(), "Inplace multiplication by constant");
        m_class->def("__imul__", (inplace_array_op)&array_type::operator*=, boost::python::return_self<>(), "Inplace multiplication by array, elementwise.");
        m_class->def("__idiv__", (inplace_const_op)&array_type::operator/=, boost::python::return_self<>(), "Inplace division by constant");
        m_class->def("__idiv__", (inplace_array_op)&array_type::operator/=, boost::python::return_self<>(), "Inplace division by array, elementwise.");
      }

      void load_transformers() {
        switch(N) { 
          case 2: 
            m_class->def("transpose", (array_type (array_type::*)(int,int))&array_type::transpose, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
            m_class->def("transposeSelf", (void (array_type::*)(int,int))&array_type::transposeSelf, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
            break;
          case 3: 
            m_class->def("transpose", (array_type (array_type::*)(int,int,int))&array_type::transpose, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
            m_class->def("transposeSelf", (void (array_type::*)(int,int,int))&array_type::transposeSelf, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
            break;
          case 4: 
            m_class->def("transpose", (array_type (array_type::*)(int,int,int,int))&array_type::transpose, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3"), boost::python::arg("dim4")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
            m_class->def("transposeSelf", (void (array_type::*)(int,int,int,int))&array_type::transposeSelf, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3"), boost::python::arg("dim4")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
            break;
        }

        m_class->def("reverse", &array_type::reverse, (boost::python::arg("self"), boost::python::arg("dimension")), "This method reverses the array in the specified dimension. For example, if reverse(firstDim) is invoked on a 2-dimensional array, then the ordering of rows in the array will be reversed; reverse(secondDim) would reverse the order of the columns. Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
        m_class->def("reverseSelf", &array_type::reverseSelf, (boost::python::arg("self"), boost::python::arg("dimension")), "This method reverses the array in the specified dimension. For example, if reverse(firstDim) is invoked on a 2-dimensional array, then the ordering of rows in the array will be reversed; reverse(secondDim) would reverse the order of the columns. Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
      }

      /**
       * Load casts to different base types
       */
      template <typename T2> void load_cast(const std::string name) {
        boost::format s("as_%s");
        s % name;
        m_class->def(s.str().c_str(), &cast<T,N,T2>, (boost::python::arg("self")), "Runs the blitz::cast() operator on the current array.");
      }

      /**
       * Loads stuff that is exclusive for specific array types
       */
      void load_bool_methods() {
      }

      void load_int_methods() {
      }

      void load_uint_methods() {
      }

      void load_float_methods() {
        m_class->def("ceil", &ceil<T,N>, "Ceiling function: smallest floating-point integer value not less than the argument."); 
        m_class->def("floor", &floor<T,N>, "Floor function: largest floating-point integer value not greater than the argument.");
      }

      void load_int_float_complex_methods() {
        m_class->def("__abs__", &__abs__<T,N>, "Absolute value");
        m_class->def("cos", &cos<T,N>, "Cosine, element-wise");
        m_class->def("cosh", &cosh<T,N>, "Hyperbolic cosine, element-wise");
        m_class->def("log", &log<T,N>, "Natural logarithm, element-wise");
        m_class->def("log10", &log10<T,N>, "Base 10 logarithm, element-wise");
        m_class->def("sin", &sin<T,N>, "Sine, element-wise");
        m_class->def("sinh", &sinh<T,N>, "Hyperbolic sine, element-wise");
        m_class->def("sqr", &sqr<T,N>, "self ** 2, element-wise");
        m_class->def("sqrt", &sqrt<T,N>, "self ** 0.5, element-wise");
        m_class->def("tan", &tan<T,N>, "Tangent, element-wise");
        m_class->def("tanh", &tanh<T,N>, "Hyperbolic tangent, element-wise");
        //TODO: m_class->def("isnan", &isnan<T,N>, "Returns a nonzero integer if the parameter is NaNQ or NaNS (quiet or signalling Not a Number), element-wise.");
      }

      void load_float_complex_methods() {
        m_class->def("__pow__", &__pow__<T,N>, "Computes self**argument.");
        m_class->def("__pow__", &__pow_c__<T,N>, "Computes self**argument.");
      }

      /**
       * Some stuff that is multi-type
       */
      void load_bool_int_uint_methods() {
        m_class->def("__and__", &__and__<T,N>, "Performs a bitwise and operation on two arrays, element-wise."); 
        m_class->def("__and__", &__and_c__<T,N>, "Performs a bitwise and operation on two arrays, element-wise."); 
        m_class->def("__or__", &__or__<T,N>, "Performs a bitwise or operation on two arrays, element-wise."); 
        m_class->def("__or__", &__or_c__<T,N>, "Performs a bitwise or operation on two arrays, element-wise."); 
        m_class->def("__xor__", &__xor__<T,N>, "Performs a bitwise xor operation on two arrays, element-wise."); 
        m_class->def("__xor__", &__xor_c__<T,N>, "Performs a bitwise xor operation on two arrays, element-wise."); 
        m_class->def("__ilshift__", (inplace_const_op)&array_type::operator<<=, boost::python::return_self<>(), "Inplace bitwise left-shift by constant.");
        m_class->def("__ilshift__", (inplace_array_op)&array_type::operator<<=, boost::python::return_self<>(), "Inplace bitwise left-shift by array, elementwise.");
        m_class->def("__irshift__", (inplace_const_op)&array_type::operator>>=, boost::python::return_self<>(), "Inplace bitwise right-shift by constant.");
        m_class->def("__irshift__", (inplace_array_op)&array_type::operator>>=, boost::python::return_self<>(), "Inplace bitwise right-shift by array, elementwise.");
        m_class->def("__iand__", (inplace_const_op)&array_type::operator&=, boost::python::return_self<>(), "Inplace bitwise and operation with constant.");
        m_class->def("__iand__", (inplace_array_op)&array_type::operator&=, boost::python::return_self<>(), "Inplace bitwise and operation with array, elementwise.");
        m_class->def("__ior__", (inplace_const_op)&array_type::operator|=, boost::python::return_self<>(), "Inplace bitwise or operation with constant.");
        m_class->def("__ior__", (inplace_array_op)&array_type::operator|=, boost::python::return_self<>(), "Inplace bitwise or operation with array, elementwise.");
        m_class->def("__ixor__", (inplace_const_op)&array_type::operator^=, boost::python::return_self<>(), "Inplace bitwise xor operation with constant.");
        m_class->def("__ixor__", (inplace_array_op)&array_type::operator^=, boost::python::return_self<>(), "Inplace bitwise xor operation with array, elementwise.");
        m_class->def("__invert__", &__invert__<T,N>, "The inverted values of the array element-wise"); 
        m_class->def("__mod__", &__mod__<T,N>, "Executes the reminder of division between two arrays, element-wise.");
        m_class->def("__mod__", &__mod_c__<T,N>, "Executes the reminder of division between two arrays, element-wise.");
        m_class->def("__imod__", (inplace_const_op)&array_type::operator%=, boost::python::return_self<>(), "Inplace reminder of division by constant");
        m_class->def("__imod__", (inplace_array_op)&array_type::operator%=, boost::python::return_self<>(), "Inplace reminder division by array, elementwise.");
      }

      void load_non_complex_methods() {
        m_class->def("__lt__", &__lt__<T,N>, "Compares two arrays element-wise"); 
        m_class->def("__lt__", &__lt_c__<T,N>, "Compares an array to a constant element-wise"); 
        m_class->def("__le__", &__le__<T,N>, "Compares two arrays element-wise"); 
        m_class->def("__le__", &__le_c__<T,N>, "Compares an array to a constant element-wise"); 
        m_class->def("__eq__", &__eq__<T,N>, "Compares two arrays element-wise"); 
        m_class->def("__eq__", &__eq_c__<T,N>, "Compares an array to a constant element-wise"); 
        m_class->def("__ne__", &__ne__<T,N>, "Compares two arrays element-wise"); 
        m_class->def("__ne__", &__ne_c__<T,N>, "Compares an array to a constant element-wise"); 
        m_class->def("__gt__", &__gt__<T,N>, "Compares two arrays element-wise"); 
        m_class->def("__gt__", &__gt_c__<T,N>, "Compares an array to a constant element-wise"); 
        m_class->def("__ge__", &__ge__<T,N>, "Compares two arrays element-wise"); 
        m_class->def("__ge__", &__ge_c__<T,N>, "Compares an array to a constant element-wise"); 
        m_class->def("__neg__", &__neg__<T,N>, "The negated values of the array element-wise"); 

        m_class->def("acosh", &acosh<T,N>, "Inverse hyperbolic cosine, element-wise");
        m_class->def("acos", &acos<T,N>, "Arc cosine, element-wise");
        m_class->def("asin", &asin<T,N>, "Arc sine, element-wise");
        m_class->def("atan", &atan<T,N>, "Arc tangent, element-wise");
        m_class->def("atanh", &atanh<T,N>, "Inverse hyperbolic tangent, element-wise");
        m_class->def("cbrt", &cbrt<T,N>, "self ** (1/3) (cubic root), element-wise");
        m_class->def("exp", &exp<T,N>, "exponential element-wise");
        m_class->def("expm1", &expm1<T,N>, "exp(x)-1, element-wise");
        m_class->def("erf", &erf<T,N>, "Computes the error function: erf(x) = 2/sqrt(Pi) * integral(exp(-t^2), t=0..x) Note that for large values of the parameter, calculating 1.0-erf(x) can result in extreme loss of accuracy. Instead, use erfc(), element-wise");
        m_class->def("erfc", &erfc<T,N>, "Computes the complementary error function erfc(x) = 1.0 - erf(x), element-wise.");
        m_class->def("ilogb", &ilogb<T,N>, "Returns an integer which is equal to the unbiased exponent of the parameter, element-wise.");
        m_class->def("rint", &rint<T,N>, "Rounds the parameter and returns a floating-point integer value. Whether rint() rounds up or down or to the nearest integer depends on the current floating-point rounding mode. If you haven't altered the rounding mode, rint() should be equivalent to nearest(). If rounding mode is set to round towards +INF, rint() is equivalent to ceil(). If the mode is round toward -INF, rint() is equivalent to floor(). If the mode is round toward zero, rint() is equivalent to trunc(). )");
        m_class->def("j0", &j0<T,N>, "Bessel function of the first kind, order 0."); 
        m_class->def("j1", &j1<T,N>, "Bessel function of the first kind, order 1."); 
        m_class->def("lgamma", &lgamma<T,N>, "Natural logarithm of the gamma function. The gamma function Gamma(x) is defined as: Gamma(x) = integral(e^(-t) * t^(x-1), t=0..infinity)."); 
        m_class->def("log1p", &log1p<T,N>, "Calculates log(1+x), where x is the parameter.");
        m_class->def("y0", &y0<T,N>, "Bessel function of the second kind, order 0.");
        m_class->def("y1", &y1<T,N>, "Bessel function of the second kind, order 1.");
        m_class->def("mean", &mean<T,N>, "Arithmetic mean");
        m_class->def("min", &Torch::python::min<T,N>, "Minimum value");
        m_class->def("max", &Torch::python::max<T,N>, "Maximum value");
        m_class->def("minIndex", &minIndex<T,N>, "Index of the minimum value (returns tuple.");
        m_class->def("maxIndex", &maxIndex<T,N>, "Index of the maximum value (returns tuple.");
        m_class->def("count", &count<T,N>, "Counts the number of times the expression is true anywhere.");
        m_class->def("any", &any<T,N>, "True if the array is True anywhere.");
        m_class->def("all", &all<T,N>, "True if the array is True everywhere.");

        //TODO: Fix footprint of object files once the stuff bellow is
        //uncommented!
        //load_cast<bool>("bool");
        //load_cast<int8_t>("int8");
        //load_cast<int16_t>("int16");
        //load_cast<int32_t>("int32");
        //load_cast<int64_t>("int64");
        //load_cast<uint8_t>("uint8");
        //load_cast<uint16_t>("uint16");
        //load_cast<uint32_t>("uint32");
        //load_cast<uint64_t>("uint64");
        //load_cast<float>("float32");
        //load_cast<double>("float64");
        //load_cast<long double>("float128");
      }

      void load_complex_methods() {
        m_class->def("real", &blitz::real<typename T::value_type,N>, "Returns the real portion of the array (reference).");
        m_class->def("imag", &blitz::imag<typename T::value_type,N>, "Returns the imag portion of the array (reference).");
        m_class->def("arg", &arg<T,N>, "Argument of a complex number (atan2(Im,Re)).");
        m_class->def("conj", conj<T,N>, "Conjugate of a complex number.");

        //TODO: See comments on the TODO before this one.
        //some casts: too much memory!!
        //load_cast<std::complex<float> >("complex64");
        //load_cast<std::complex<double> >("complex128");
        //load_cast<std::complex<long double> >("complex256");
      }

      /**
       * This will load mathematical functions that operate on blitz arrays
       */
      void load_math_operators() {  //multi-T operands
        m_class->def("sum", &sum<T,N>, "Summation");
        m_class->def("product", &product<T,N>, "Product");
        m_class->def("fill", &fill<T,N>, (boost::python::arg("self"), boost::python::arg("value")), "Fills the array with the same value all over");
        m_class->def("zeroes", &zeroes<T,N>, "Fills the array with zeroes");
        m_class->def("ones", &ones<T,N>, "Fills the array with ones");
      }

  };

  template<typename T, int N> std::string array_binder<T,N>::s_element_type_str("unknown");
  template<typename T, int N> std::string array_binder<T,N>::s_blitz_type_str("blitz::Array<unknown,?>");

} } //namespace Torch::python 

/**
 * Current extra methods available / type applicability
 * - load_bool_methods / bool
 * - load_int_methods / int
 * - load_uint_methods / uint
 * - load_float_methods / float
 * - load_complex_methods / complex
 * - load_bool_int_uint_methods / bool, int, uint
 * - load_non_complex_methods / bool, int, uint, float
 * - load_int_float_complex_methods / int, float, complex
 * - load_float_complex_methods / float, complex
 */

#define declare_bool_array(T,D,NAME,FNAME) void FNAME() { \
  Torch::python::array_binder<T,D> a(BOOST_STRINGIZE(NAME)); \
  a.load_bool_methods(); \
  a.load_bool_int_uint_methods(); \
  a.load_non_complex_methods(); \
}

#define declare_integer_array(T,D,NAME,FNAME) void FNAME() { \
  Torch::python::array_binder<T,D> a(BOOST_STRINGIZE(NAME)); \
  a.load_int_methods(); \
  a.load_bool_int_uint_methods(); \
  a.load_non_complex_methods(); \
  a.load_int_float_complex_methods(); \
}

#define declare_unsigned_array(T,D,NAME,FNAME) void FNAME() { \
  Torch::python::array_binder<T,D> a(BOOST_STRINGIZE(NAME)); \
  a.load_uint_methods(); \
  a.load_bool_int_uint_methods(); \
  a.load_non_complex_methods(); \
}

#define declare_float_array(T,D,NAME,FNAME) void FNAME() { \
  Torch::python::array_binder<T,D> a(BOOST_STRINGIZE(NAME)); \
  a.load_float_methods(); \
  a.load_non_complex_methods(); \
  a.load_int_float_complex_methods(); \
  a.load_float_complex_methods(); \
}

#define declare_complex_array(T,D,NAME,FNAME) void FNAME() { \
  Torch::python::array_binder<T,D> a(BOOST_STRINGIZE(NAME)); \
  a.load_complex_methods(); \
  a.load_int_float_complex_methods(); \
  a.load_float_complex_methods(); \
}

#endif //TORCH_CORE_PYTHON_ARRAY
