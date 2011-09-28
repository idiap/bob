/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon 14 Mar 09:53:06 2011 
 *
 * @brief Several methods allowing conversion to/from numpy 
 */

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#define PY_ARRAY_UNIQUE_SYMBOL torch_NUMPY_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <boost/format.hpp>
#include <blitz/array.h>

#include "core/cast.h"
#include "core/python/bzhelper.h"
#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"

namespace tp = Torch::python;
namespace bp = boost::python;
namespace tca = Torch::core::array;

inline static PyArrayObject* make_new_array(int nd, npy_intp* dims, int type) {
  return (PyArrayObject*)PyArray_SimpleNew(nd, dims, type);
}

template <typename T, int N> static
PyObject* make_ndarray(const blitz::Array<T,N>& tv) {

  typedef typename blitz::Array<T,N> array_type;
  typedef typename blitz::TinyVector<int,N> shape_type;

  npy_intp dims[N];
  for (int i=0; i<N; ++i) dims[i] = tv.extent(i);
  PyArrayObject* retval = make_new_array(N, dims, tp::type_to_num<T>());

  //wrap new PyArray in a blitz layer and then copy the data
  shape_type shape;
  for (int k=0; k<retval->nd; ++k) shape[k] = retval->dimensions[k];
  shape_type stride;
  for (int k=0; k<retval->nd; ++k) stride[k] = (retval->strides[k]/sizeof(T));
  array_type bzdest((T*)retval->data, shape, stride, blitz::neverDeleteData);
  bzdest = tv;

  return reinterpret_cast<PyObject*>(retval);
}

/**
 * Load casts to different base types
 */
template <typename T, int N> static 
bp::object cast(blitz::Array<T,N>& array, const char* eltype_string) {
  tca::ElementType eltype = tca::unstringize(eltype_string);
  switch (eltype) {
    case tca::t_bool: 
      return bp::object(Torch::core::cast<bool,T>(array));
    case tca::t_int8: 
      return bp::object(Torch::core::cast<int8_t,T>(array));
    case tca::t_int16: 
      return bp::object(Torch::core::cast<int16_t,T>(array));
    case tca::t_int32: 
      return bp::object(Torch::core::cast<int32_t,T>(array));
    case tca::t_int64: 
      return bp::object(Torch::core::cast<int64_t,T>(array));
    case tca::t_uint8: 
      return bp::object(Torch::core::cast<uint8_t,T>(array));
    case tca::t_uint16: 
      return bp::object(Torch::core::cast<uint16_t,T>(array));
    case tca::t_uint32: 
      return bp::object(Torch::core::cast<uint32_t,T>(array));
    case tca::t_uint64: 
      return bp::object(Torch::core::cast<uint64_t,T>(array));
    case tca::t_float32: 
      return bp::object(Torch::core::cast<float,T>(array));
    case tca::t_float64: 
      return bp::object(Torch::core::cast<double,T>(array));
    //case tca::t_float128: 
    //  return bp::object(Torch::core::cast<long double,T>(array));
    case tca::t_complex64: 
      return bp::object(Torch::core::cast<std::complex<float>,T>(array));
    case tca::t_complex128: 
      return bp::object(Torch::core::cast<std::complex<double>,T>(array));
    //case tca::t_complex256: 
    //  return bp::object(Torch::core::cast<std::complex<long double>,T>(array));
    default:
      break;
  }
  boost::format msg("Unsupported cast type: %s (interpreted as %s)");
  msg % eltype_string;
  msg % tca::stringize(eltype);
  PyErr_SetString(PyExc_TypeError, msg.str().c_str());
  bp::throw_error_already_set();
  return bp::object(); //shuts up gcc
}

/**
 * Loads stuff for exporting the array
 */
template <typename T, int N> static void bind_cast (tp::array<T,N>& array) {
  array.object()->def("as_ndarray", &make_ndarray<T,N>, (bp::arg("self")), "Creates a copy of this array as a NumPy Array with the same dimensions and storage type");
  array.object()->def("__array__", &make_ndarray<T,N>, (bp::arg("self")), "Creates a copy of this array as a NumPy Array with the same dimensions and storage type");
  array.object()->def("cast", &cast<T,N>, (bp::arg("self"), bp::arg("cast_to")), "Creates a copy of this array as another blitz::Array<> with different element type. The cast_to parameter should be picked from one of the allowed values for Torch blitz Arrays");
}

void bind_array_cast () {
# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) \
  bind_cast(BOOST_PP_CAT(tp::bool_,D));\
  bind_cast(BOOST_PP_CAT(tp::int8_,D));\
  bind_cast(BOOST_PP_CAT(tp::int16_,D));\
  bind_cast(BOOST_PP_CAT(tp::int32_,D));\
  bind_cast(BOOST_PP_CAT(tp::int64_,D));\
  bind_cast(BOOST_PP_CAT(tp::uint8_,D));\
  bind_cast(BOOST_PP_CAT(tp::uint16_,D));\
  bind_cast(BOOST_PP_CAT(tp::uint32_,D));\
  bind_cast(BOOST_PP_CAT(tp::uint64_,D));\
  bind_cast(BOOST_PP_CAT(tp::float32_,D));\
  bind_cast(BOOST_PP_CAT(tp::float64_,D));\
  bind_cast(BOOST_PP_CAT(tp::complex64_,D));\
  bind_cast(BOOST_PP_CAT(tp::complex128_,D));
# include BOOST_PP_LOCAL_ITERATE()
}
