/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 27 Sep 12:36:44 2011
 *
 * @brief Implementation of a few blitz::Array<> => numpy.ndarray helpers.
 */

#define torch_IMPORT_ARRAY
#include "core/python/pycore.h"
#undef torch_IMPORT_ARRAY
#include <boost/format.hpp>

namespace bp = boost::python;
namespace tp = Torch::python;

void tp::setup_python(const char* module_docstring) {

  // Documentation options
  bp::docstring_options docopt;
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  if (module_docstring) bp::scope().attr("__doc__") = module_docstring;

  // Gets the current dlopenflags and save it
  PyThreadState *tstate = PyThreadState_GET();
  if(!tstate) throw std::runtime_error("Can not get python dlopenflags.");
  int old_value = tstate->interp->dlopenflags;

  // Unsets the RTLD_GLOBAL flag
  tstate->interp->dlopenflags = old_value & (~RTLD_GLOBAL);

  // Loads numpy with the RTLD_GLOBAL flag unset
  import_array();

  // Resets the RTLD_GLOBAL flag
  tstate->interp->dlopenflags = old_value;

  //Sets the boost::python::numeric::array interface to use numpy.ndarray
  //as basis. This is not strictly required, but good to set as a baseline.
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
}

int tp::eltype_to_num(Torch::core::array::ElementType eltype) {
  switch(eltype) {
    case Torch::core::array::t_bool:
      return NPY_BOOL;
    case Torch::core::array::t_int8:
      return NPY_BYTE;
    case Torch::core::array::t_int16:
      return NPY_SHORT;
    case Torch::core::array::t_int32:
      return NPY_INT;
    case Torch::core::array::t_int64:
      return (sizeof(long) == 8)?NPY_LONG:NPY_LONGLONG;
    case Torch::core::array::t_uint8:
      return NPY_UBYTE;
    case Torch::core::array::t_uint16:
      return NPY_USHORT;
    case Torch::core::array::t_uint32:
      return NPY_UINT;
    case Torch::core::array::t_uint64:
      return (sizeof(unsigned long) == 8)?NPY_ULONG:NPY_ULONGLONG;
    case Torch::core::array::t_float32:
      return NPY_FLOAT;
    case Torch::core::array::t_float64:
      return NPY_DOUBLE;
    case Torch::core::array::t_float128:
      return NPY_LONGDOUBLE;
    case Torch::core::array::t_complex64:
      return NPY_CFLOAT;
    case Torch::core::array::t_complex128:
      return NPY_CDOUBLE;
    case Torch::core::array::t_complex256:
      return NPY_CLONGDOUBLE;
    default:
      PYTHON_ERROR(TypeError, "unsupported C++ element type -- debug me!");
  }
}

Torch::core::array::ElementType tp::num_to_eltype(int num) {
  switch(num) {
    case NPY_BOOL:
      return Torch::core::array::t_uint8;
    case NPY_BYTE:
      return Torch::core::array::t_int8;
    case NPY_SHORT:
      return Torch::core::array::t_int16;
    case NPY_INT:
      return Torch::core::array::t_int32;
    case NPY_LONG:
      return (sizeof(long) == 8)?Torch::core::array::t_int64:Torch::core::array::t_int32;
    case NPY_LONGLONG:
      return Torch::core::array::t_int64;
    case NPY_UBYTE:
      return Torch::core::array::t_uint8;
    case NPY_USHORT:
      return Torch::core::array::t_uint16;
    case NPY_UINT:
      return Torch::core::array::t_uint32;
    case NPY_ULONG:
      return (sizeof(unsigned long) == 8)?Torch::core::array::t_uint64:Torch::core::array::t_uint32;
    case NPY_ULONGLONG:
      return Torch::core::array::t_uint64;
    case NPY_FLOAT:
      return Torch::core::array::t_float32;
    case NPY_DOUBLE:
      return Torch::core::array::t_float64;
    case NPY_LONGDOUBLE:
      return Torch::core::array::t_float128;
    case NPY_CFLOAT:
      return Torch::core::array::t_complex64;
    case NPY_CDOUBLE:
      return Torch::core::array::t_complex128;
    case NPY_CLONGDOUBLE:
      return Torch::core::array::t_complex256;
    default:
      PYTHON_ERROR(TypeError, "unsupported NumPy element type -- debug me!");
  }
}

template <> int tp::type_to_num<bool>(void) 
{ return NPY_BOOL; }
template <> int tp::type_to_num<signed char>(void) 
{ return NPY_BYTE; }
template <> int tp::type_to_num<unsigned char>(void) 
{ return NPY_UBYTE; }
template <> int tp::type_to_num<short>(void) 
{ return NPY_SHORT; }
template <> int tp::type_to_num<unsigned short>(void) 
{ return NPY_USHORT; }
template <> int tp::type_to_num<int>(void) 
{ return NPY_INT; }
template <> int tp::type_to_num<unsigned int>(void) 
{ return NPY_UINT; }
template <> int tp::type_to_num<long>(void)
{ return NPY_LONG; }
template <> int tp::type_to_num<unsigned long>(void)
{ return NPY_ULONG; }
template <> int tp::type_to_num<long long>(void)
{ return NPY_LONGLONG; }
template <> int tp::type_to_num<unsigned long long>(void)
{ return NPY_ULONGLONG; }
template <> int tp::type_to_num<float>(void)
{ return NPY_FLOAT; }
template <> int tp::type_to_num<double>(void) 
{ return NPY_DOUBLE; }
template <> int tp::type_to_num<long double>(void) 
{ return NPY_LONGDOUBLE; }
template <> int tp::type_to_num<std::complex<float> >(void)
{ return NPY_CFLOAT; }
template <> int tp::type_to_num<std::complex<double> >(void) 
{ return NPY_CDOUBLE; }
template <> int tp::type_to_num<std::complex<long double> >(void) 
{ return NPY_CLONGDOUBLE; }

void tp::assert_ndarray_shape(PyArrayObject* arr, int N) {
  if (arr->nd != N) {
    boost::format f("shape mismatch in ndarray(dtype='%c%c%d',nd=%d) => blitz::Array<?,%d> wrapping");
    f % arr->descr->byteorder % arr->descr->kind % arr->descr->elsize % arr->nd % N;
    PYTHON_ERROR(TypeError, f.str().c_str());
  }
}

void tp::assert_ndarray_type(PyArrayObject* arr, int type_num) {
  if (type_num != arr->descr->type_num) {
    boost::format f("type mismatch in ndarray(dtype='%c%c%d',nd=%d) => blitz::Array<%c%c%d,?> wrapping");
    PyArray_Descr* bzdescr = tp::describe_ndarray(type_num);
    f % arr->descr->byteorder % arr->descr->kind % arr->descr->elsize % arr->nd % bzdescr->byteorder % bzdescr->kind % bzdescr->elsize;
    PYTHON_ERROR(TypeError, f.str().c_str());
  }
}

void tp::assert_ndarray_byteorder(PyArrayObject* arr) {
  if (!PyArray_EquivByteorders(arr->descr->byteorder, NPY_NATIVE)) {
    boost::format f("cannot digest non-native byte ordering in ndarray(dtype='%c%c%d',nd=%d) => blitz::Array wrapping");
    f % arr->descr->byteorder % arr->descr->kind % arr->descr->elsize % arr->nd;
    PYTHON_ERROR(TypeError, f.str().c_str());
  }
}

void tp::assert_ndarray_writeable(PyArrayObject* arr) {
  if (!PyArray_ISWRITEABLE(arr)) {
    boost::format f("cannot apply blitz layer on const ndarray in ndarray(dtype='%c%c%d',nd=%d) => blitz::Array wrapping");
    f % arr->descr->byteorder % arr->descr->kind % arr->descr->elsize % arr->nd;
    PYTHON_ERROR(TypeError, f.str().c_str());
  }
}

void tp::assert_ndarray_behaved(PyArrayObject* arr) {
  if (!PyArray_ISBEHAVED(arr)) {
    boost::format f("cannot apply blitz layer on ndarray that is not well-behaved (search for what this means with PyArray_ISBEHAVE) in ndarray(dtype='%c%c%d',nd=%d) => blitz::Array wrapping");
    f % arr->descr->byteorder % arr->descr->kind % arr->descr->elsize % arr->nd;
    PYTHON_ERROR(TypeError, f.str().c_str());
  }
}

tp::dtype::dtype(const bp::object& name): _m(0) {
  PyArray_DescrConverter(name.ptr(), &_m);
}

tp::dtype::dtype(const tp::dtype& other): _m(other._m) {
}

tp::dtype::~dtype() {
}

PyArrayObject* tp::make_ndarray(int nd, npy_intp* dims, int type) {
  return (PyArrayObject*)PyArray_SimpleNew(nd, dims, type);
}

PyArray_Descr* tp::describe_ndarray(int type) {
  return PyArray_DescrFromType(type);
}

PyArrayObject* tp::copy_ndarray(PyObject* any, PyArray_Descr* dt,
    int dims) { 
  PyArrayObject* retval = (PyArrayObject*)PyArray_FromAny(any, dt, dims, dims,
#if C_API_VERSION >= 6 /* NumPy C-API version > 1.6 */
      NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_ENSURECOPY
#else
      NPY_C_CONTIGUOUS|NPY_ALIGNED|NPY_ENSURECOPY
#endif
  ,0);
  if (!retval) {
    bp::throw_error_already_set();
  }
  return retval;
}

int tp::check_ndarray(PyObject* any, PyArray_Descr* req_dtype,
    int writeable, PyArray_Descr*& dtype, int& ndim, npy_intp* dims,
    PyArrayObject*& arr) {
#if C_API_VERSION >= 6 /* NumPy C-API version < 1.6 */
  return PyArray_GetArrayParamsFromObject(
      any,   //input object pointer
      req_dtype, //requested dtype (if need to enforce)
      0,         //writeable?
      &dtype,    //dtype assessment
      &ndim,     //assessed number of dimensions
      dims,     //assessed shape
      &arr,     //if obj_ptr is ndarray, return it here
      NULL)      //context?
    ;
#else
  //well, in this case we have to implement the above manually.
  //for sequence conversions, this will be a little more inefficient.
  if (PyArray_Check(any)) {
    PyArray_Descr* descr = PyArray_DESCR(any);
    if (req_dtype) {
      if (req_dtype->type_num == descr->type_num) {
        arr = (PyArrayObject*)any;
      }
      else { //fill the other variables
        dtype = descr;
        ndim = ((PyArrayObject*)any)->nd;
        for (int k=0; k<ndim; ++k) dims[k] = PyArray_DIM(any, k);
      }
    }
    else { //the user has not requested a specific type, just return
      arr = (PyArrayObject*)any;
    }
  }
  else { //it is not an array -- try a conversion
    PyArrayObject* tmp = tp::copy_ndarray(any, 0, 0);
    dtype = tmp->descr;
    ndim = tmp->nd;
    for (int k=0; k<ndim; ++k) dims[k] = PyArray_DIM(tmp, k);
    Py_DECREF(tmp);
  }
  return 0;
#endif
}
