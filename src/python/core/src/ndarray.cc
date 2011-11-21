/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 21 Nov 2011 12:10:13 CET
 *
 * @brief Implementation of the ndarray class
 */

#include <boost/python/numeric.hpp>
#include <boost/format.hpp>
#include <stdexcept>
#include <dlfcn.h>

#define torch_IMPORT_ARRAY
#include "core/python/ndarray.h"
#undef torch_IMPORT_ARRAY

#include "core/logging.h"

namespace bp = boost::python;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

#define TP_ARRAY(x) ((PyArrayObject*)x.ptr())
#define TP_OBJECT(x) (x.ptr())

#define NUMPY16_API 0x00000006
#define NUMPY14_API 0x00000004

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

  //Sets the bp::numeric::array interface to use numpy.ndarray
  //as basis. This is not strictly required, but good to set as a baseline.
  bp::numeric::array::set_module_and_type("numpy", "ndarray");

  // Make sure we are not running against the wrong version of NumPy
  if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {
    PYTHON_ERROR(ImportError, "module compiled against ABI version 0x%08x but this version of numpy is 0x%08x - make sure you compile and execute against the same or compatible versions", (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
  }

#if NPY_FEATURE_VERSION >= NUMPY14_API /* NumPy C-API version >= 1.4 */
  if (NPY_FEATURE_VERSION > PyArray_GetNDArrayCFeatureVersion()) {
    PYTHON_ERROR(ImportError, "module compiled against API version 0x%08x but this version of numpy is 0x%08x - make sure you compile and execute against the same or compatible versions", (int) NPY_FEATURE_VERSION, (int) PyArray_GetNDArrayCFeatureVersion());
  }
#endif

}

/***************************************************************************
 * Dtype (PyArray_Descr) manipulations                                     *
 ***************************************************************************/

bp::object tp::make_non_null_object(PyObject* obj) {
  bp::handle<> hdl(obj); //< raises if NULL
  bp::object retval(hdl);
  return retval;
}

bp::object tp::make_maybe_null_object(PyObject* obj) {
  bp::handle<> hdl(bp::allow_null(obj));
  bp::object retval(hdl);
  return retval;
}

bp::object tp::make_non_null_borrowed_object(PyObject* obj) {
  bp::handle<> hdl(bp::borrowed(obj));
  bp::object retval(hdl);
  return retval;
}

int tp::type_to_num(ca::ElementType type) {

  switch(type) {
    case ca::t_bool:
      return NPY_BOOL;
    case ca::t_int8:
      return NPY_INT8;
    case ca::t_int16:
      return NPY_INT16;
    case ca::t_int32:
      return NPY_INT32;
    case ca::t_int64:
      return NPY_INT64;
    case ca::t_uint8:
      return NPY_UINT8;
    case ca::t_uint16:
      return NPY_UINT16;
    case ca::t_uint32:
      return NPY_UINT32;
    case ca::t_uint64:
      return NPY_UINT64;
    case ca::t_float32:
      return NPY_FLOAT32;
    case ca::t_float64:
      return NPY_FLOAT64;
    case ca::t_float128:
      return NPY_FLOAT128;
    case ca::t_complex64:
      return NPY_COMPLEX64;
    case ca::t_complex128:
      return NPY_COMPLEX128;
    case ca::t_complex256:
      return NPY_COMPLEX256;
    default:
      PYTHON_ERROR(TypeError, "unsupported C++ element type -- debug me!");
  }

}

ca::ElementType tp::num_to_type(int num) {
  switch(num) {
    case NPY_BOOL:
      return ca::t_bool;
    case NPY_INT8:
      return ca::t_int8;
    case NPY_INT16:
      return ca::t_int16;
    case NPY_INT32:
      return ca::t_int32;
    case NPY_INT64:
      return ca::t_int64;
    case NPY_UINT8:
      return ca::t_uint8;
    case NPY_UINT16:
      return ca::t_uint16;
    case NPY_UINT32:
      return ca::t_uint32;
    case NPY_UINT64:
      return ca::t_uint64;
    case NPY_FLOAT32:
      return ca::t_float32;
    case NPY_FLOAT64:
      return ca::t_float64;
    case NPY_FLOAT128:
      return ca::t_float128;
    case NPY_COMPLEX64:
      return ca::t_complex64;
    case NPY_COMPLEX128:
      return ca::t_complex128;
    case NPY_COMPLEX256:
      return ca::t_complex256;
    default:
      PYTHON_ERROR(TypeError, "unsupported NumPy element type -- debug me!");
  }

}

template <> int tp::ctype_to_num<bool>(void) 
{ return NPY_BOOL; }
template <> int tp::ctype_to_num<int8_t>(void) 
{ return NPY_INT8; }
template <> int tp::ctype_to_num<uint8_t>(void) 
{ return NPY_UINT8; }
template <> int tp::ctype_to_num<int16_t>(void) 
{ return NPY_INT16; }
template <> int tp::ctype_to_num<uint16_t>(void) 
{ return NPY_UINT16; }
template <> int tp::ctype_to_num<int32_t>(void) 
{ return NPY_INT32; }
template <> int tp::ctype_to_num<uint32_t>(void) 
{ return NPY_UINT32; }
template <> int tp::ctype_to_num<int64_t>(void)
{ return NPY_INT64; }
template <> int tp::ctype_to_num<uint64_t>(void)
{ return NPY_UINT64; }
template <> int tp::ctype_to_num<float>(void)
{ return NPY_FLOAT32; }
template <> int tp::ctype_to_num<double>(void) 
{ return NPY_FLOAT64; }
template <> int tp::ctype_to_num<long double>(void) 
{ return NPY_FLOAT128; }
template <> int tp::ctype_to_num<std::complex<float> >(void)
{ return NPY_COMPLEX64; }
template <> int tp::ctype_to_num<std::complex<double> >(void) 
{ return NPY_COMPLEX128; }
template <> int tp::ctype_to_num<std::complex<long double> >(void) 
{ return NPY_COMPLEX256; }

ca::ElementType tp::array_to_type(const bp::numeric::array& a) {
  return tp::num_to_type(TP_ARRAY(a)->descr->type_num);
}

size_t tp::array_to_ndim(const bp::numeric::array& a) {
  return PyArray_NDIM(a.ptr());
}

#define TP_DESCR(x) ((PyArray_Descr*)x.ptr())

tp::dtype::dtype (bp::object dtype_like) {
  PyArray_Descr* tmp = 0;
  if (!PyArray_DescrConverter2(dtype_like.ptr(), &tmp)) {
    std::string dtype_str = bp::extract<std::string>(bp::str(dtype_like));
    PYTHON_ERROR(TypeError, "cannot convert input dtype-like object (%s) to proper dtype", dtype_str.c_str());
  }
  m_self = tp::make_non_null_borrowed_object((PyObject*)tmp);
}

tp::dtype::dtype (PyArray_Descr* descr) {
  if (descr) m_self = tp::make_non_null_object((PyObject*)descr);
}

tp::dtype::dtype(int typenum) {
  PyArray_Descr* tmp = PyArray_DescrFromType(typenum);
  m_self = tp::make_non_null_borrowed_object((PyObject*)tmp);
}

tp::dtype::dtype(ca::ElementType eltype) {
  if (eltype != ca::t_unknown) {
    PyArray_Descr* tmp = PyArray_DescrFromType(tp::type_to_num(eltype));
    m_self = tp::make_non_null_borrowed_object((PyObject*)tmp);
  }
}

tp::dtype::dtype(const tp::dtype& other): m_self(other.m_self)
{
}

tp::dtype::dtype() {
}

tp::dtype::~dtype() { }

tp::dtype& tp::dtype::operator= (const tp::dtype& other) {
  m_self = other.m_self;
  return *this;
}

bool tp::dtype::has_native_byteorder() const {
  return TPY_ISNONE(m_self)? false : (PyArray_EquivByteorders(TP_DESCR(m_self)->byteorder, NPY_NATIVE) || TP_DESCR(m_self)->elsize == 1);
}

bool tp::dtype::has_type(ca::ElementType _eltype) const {
  return eltype() == _eltype;
}

ca::ElementType tp::dtype::eltype() const {
  return TPY_ISNONE(m_self)? ca::t_unknown : 
    tp::num_to_type(TP_DESCR(m_self)->type_num);
}
      
int tp::dtype::type_num() const {
  return TPY_ISNONE(m_self)? -1 : TP_DESCR(m_self)->type_num;
}

bp::str tp::dtype::str() const {
  return bp::str(m_self);
}

std::string tp::dtype::cxx_str() const {
  return bp::extract<std::string>(this->str());
}

/****************************************************************************
 * Free methods                                                             *
 ****************************************************************************/

void tp::typeinfo_ndarray_ (const bp::object& o, ca::typeinfo& i) {
  PyArrayObject* npy = TP_ARRAY(o);
  npy_intp strides[NPY_MAXDIMS];
  for (int k=0; k<npy->nd; ++k) strides[k] = npy->strides[k]/npy->descr->elsize;
  i.set<npy_intp>(tp::num_to_type(npy->descr->type_num), npy->nd,
      npy->dimensions, strides);
}

void tp::typeinfo_ndarray (const bp::object& o, ca::typeinfo& i) {
  if (!PyArray_Check(o.ptr())) {
    throw std::invalid_argument("invalid input: cannot extract typeinfo object from anything else than ndarray");
  }
  tp::typeinfo_ndarray_(o, i);
}

/**
 * This method emulates the behavior of PyArray_GetArrayParamsFromObject from
 * NumPy >= 1.6 and is used when compiling and liking against older versions of
 * NumPy.
 */
static int _GetArrayParamsFromObject(PyObject* op, 
    PyArray_Descr* requested_dtype, 
    npy_bool writeable, 
    PyArray_Descr** out_dtype, 
    int* out_ndim,
    npy_intp* out_dims, 
    PyArrayObject** out_arr, 
    PyObject*) {
  
  if (PyArray_Check(op)) { //it is already an array, easy
    
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(op);

    if (requested_dtype && !PyArray_EquivTypes(arr->descr, requested_dtype)) {

      if (PyArray_CanCastTo(arr->descr, requested_dtype)) {
        (*out_arr) = 0;
        (*out_dtype) = PyArray_DESCR(arr);
        (*out_ndim) = PyArray_NDIM(arr);
        for (int i=0; i<PyArray_NDIM(arr); ++i) 
          out_dims[i] = PyArray_DIM(arr,i);
        return writeable? PyArray_ISWRITEABLE(arr) : 1;
      }
      
      else {
        return 0;
      }

    }

    //if you get to this point, the types are equivalent or there was no type
    (*out_arr) = arr;
    (*out_dtype) = 0;
    (*out_ndim) = 0;
    return writeable? PyArray_ISWRITEABLE(arr) : 1;

  }

  else { //it is not an array -- try a brute-force conversion

    TDEBUG1("[non-optimal] using NumPy version < 1.6 requires we convert input data for convertibility check - compile against NumPy >= 1.6 to improve performance");
    bp::object array = 
      tp::make_maybe_null_object(PyArray_FromAny(op, requested_dtype, 0, 0, 0, 0));
    
    if (TPY_ISNONE(array)) return 0;

    //if the conversion worked, you can now fill in the parameters
    (*out_arr) = 0;
    (*out_dtype) = requested_dtype;
    (*out_ndim) = PyArray_NDIM(TP_ARRAY(array));
    for (int i=0; i<PyArray_NDIM(TP_ARRAY(array)); ++i) 
      out_dims[i] = PyArray_DIM(TP_ARRAY(array),i);

    //in this mode, the resulting object will never be write-able.
    return writeable? 0 : 1;

  }

  return 0; //turn-off c compiler warnings...

}

tp::convert_t tp::convertible(bp::object array_like, ca::typeinfo& info,
    bool writeable, bool behaved) {

  int ndim = 0;
  npy_intp dims[NPY_MAXDIMS];
  PyArrayObject* arr = 0;
  PyArray_Descr* dtype = 0;

  int not_convertible =
#if NPY_FEATURE_VERSION >= NUMPY16_API /* NumPy C-API version >= 1.6 */
    PyArray_GetArrayParamsFromObject
#else
    _GetArrayParamsFromObject
#endif
    (array_like.ptr(), //input object pointer
     0,                //requested dtype (if need to enforce)
     writeable,        //writeable?
     &dtype,           //dtype assessment - borrowed
     &ndim,            //assessed number of dimensions
     dims,             //assessed shape
     &arr,             //if obj_ptr is ndarray, return it here
     0)                //context?
    ;

  if (not_convertible) return tp::IMPOSSIBLE;

  convert_t retval = tp::BYREFERENCE;
    
  if (arr) { //the passed object is an array

    //checks behavior.
    if (behaved) {
      if (!(PyArray_EquivByteorders(arr->descr->byteorder, NPY_NATIVE) ||
            arr->descr->elsize == 1)) retval = tp::WITHARRAYCOPY;
      if (!PyArray_ISCARRAY_RO(arr)) retval = tp::WITHARRAYCOPY;
    }

    info.set<npy_intp>(tp::num_to_type(arr->descr->type_num),
        PyArray_NDIM(arr), PyArray_DIMS(arr));
  }

  else { //the passed object is not an array
    info.set<npy_intp>(tp::num_to_type(dtype->type_num), ndim, dims);
    retval = tp::WITHCOPY;
  }

  return retval;
}

tp::convert_t tp::convertible_to (bp::object array_like,
    const ca::typeinfo& info, bool writeable, bool behaved) {
  
  tp::dtype req_dtype(info.dtype);

  int ndim = 0;
  npy_intp dims[NPY_MAXDIMS];
  PyArrayObject* arr = 0;
  PyArray_Descr* dtype = 0;

  int not_convertible =
#if NPY_FEATURE_VERSION >= NUMPY16_API /* NumPy C-API version >= 1.6 */
    PyArray_GetArrayParamsFromObject
#else
    _GetArrayParamsFromObject
#endif
    (array_like.ptr(),           //input object pointer
     TP_DESCR(req_dtype.self()), //requested dtype (if need to enforce)
     writeable,                  //writeable?
     &dtype,                     //dtype assessment
     &ndim,                      //assessed number of dimensions
     dims,                       //assessed shape
     &arr,                       //if obj_ptr is ndarray, return it here
     0)                          //context?
    ;

  if (not_convertible) return tp::IMPOSSIBLE;

  convert_t retval = tp::BYREFERENCE;
    
  if (arr) { //the passed object is an array -- check compatibility
  
    if (info.nd) { //check number of dimensions and shape, if needs to
      if (PyArray_NDIM(arr) != (int)info.nd) return tp::IMPOSSIBLE;
      if (info.has_valid_shape())
        for (size_t i=0; i<info.nd; ++i)
          if ((int)info.shape[i] != PyArray_DIM(arr,i)) return tp::IMPOSSIBLE;
    }

    //checks behavior.
    if (behaved) {
      if (!(PyArray_EquivByteorders(arr->descr->byteorder, NPY_NATIVE) ||
            arr->descr->elsize == 1)) retval = tp::WITHARRAYCOPY;
      if (!PyArray_ISCARRAY_RO(arr)) retval = tp::WITHARRAYCOPY;
    }

    return retval;

  }

  else { //the passed object is not an array

     if (info.nd) { //check number of dimensions and shape
      if (ndim != (int)info.nd) return tp::IMPOSSIBLE;
      for (size_t i=0; i<info.nd; ++i) 
        if (info.shape[i] && 
            (int)info.shape[i] != dims[i]) return tp::WITHCOPY;
     }

     retval = tp::WITHCOPY;

  }

  return retval;
}

tp::convert_t tp::convertible_to(bp::object array_like, bp::object dtype_like,
    bool writeable, bool behaved) {

  tp::dtype req_dtype(dtype_like);

  int ndim = 0;
  npy_intp dims[NPY_MAXDIMS];
  PyArrayObject* arr = 0;
  PyArray_Descr* dtype = 0;

  int not_convertible =
#if NPY_FEATURE_VERSION >= NUMPY16_API /* NumPy C-API version >= 1.6 */
    PyArray_GetArrayParamsFromObject
#else
    _GetArrayParamsFromObject
#endif
    (array_like.ptr(),           //input object pointer
     TP_DESCR(req_dtype.self()), //requested dtype (if need to enforce)
     writeable,                  //writeable?
     &dtype,                     //dtype assessment
     &ndim,                      //assessed number of dimensions
     dims,                       //assessed shape
     &arr,                       //if obj_ptr is ndarray, return it here
     0)                          //context?
    ;

  if (not_convertible) return tp::IMPOSSIBLE;

  convert_t retval = tp::BYREFERENCE;
    
  if (arr) { //the passed object is an array -- check compatibility

    //checks behavior.
    if (behaved) {
      if (!(PyArray_EquivByteorders(arr->descr->byteorder, NPY_NATIVE) ||
            arr->descr->elsize == 1)) retval = tp::WITHARRAYCOPY;
      if (!PyArray_ISCARRAY_RO(arr)) retval = tp::WITHARRAYCOPY;
    }

  }

  else { //the passed object is not an array

     retval = tp::WITHCOPY;

  }

  return retval;
}

tp::convert_t tp::convertible_to(bp::object array_like, bool writeable,
    bool behaved) {

  int ndim = 0;
  npy_intp dims[NPY_MAXDIMS];
  PyArrayObject* arr = 0;
  PyArray_Descr* dtype = 0;

  int not_convertible =
#if NPY_FEATURE_VERSION >= NUMPY16_API /* NumPy C-API version >= 1.6 */
    PyArray_GetArrayParamsFromObject
#else
    _GetArrayParamsFromObject
#endif
    (array_like.ptr(), //input object pointer
     0,                //requested dtype (if need to enforce)
     writeable,        //writeable?
     &dtype,           //dtype assessment
     &ndim,            //assessed number of dimensions
     dims,             //assessed shape
     &arr,             //if obj_ptr is ndarray, return it here
     0)                //context?
    ;

  if (not_convertible) return tp::IMPOSSIBLE;

  convert_t retval = tp::BYREFERENCE;
    
  if (arr) { //the passed object is an array -- check compatibility

    //checks behavior.
    if (behaved) {
      if (!(PyArray_EquivByteorders(arr->descr->byteorder, NPY_NATIVE) ||
            arr->descr->elsize == 1)) retval = tp::WITHARRAYCOPY;
      if (!PyArray_ISCARRAY_RO(arr)) retval = tp::WITHARRAYCOPY;
    }

  }

  else { //the passed object is not an array

     retval = tp::WITHCOPY;

  }

  return retval;
}

/***************************************************************************
 * Ndarray (PyArrayObject) manipulations                                   *
 ***************************************************************************/

/**
 * Returns either a reference or a copy of the given array_like object,
 * depending on the following requirements for referral:
 *
 * 0. The pointed object is a numpy.ndarray
 * 1. The array type description type_num matches
 * 2. The array is C-style, contiguous and aligned
 */
static bp::object try_refer_ndarray (boost::python::object array_like, 
    bp::object dtype_like) {

  PyArrayObject* candidate = TP_ARRAY(array_like);
  PyArray_Descr* req_dtype = 0;
  PyArray_DescrConverter2(dtype_like.ptr(), &req_dtype); //new ref!

  bool can_refer = true; //< flags a copy of the data

  if (!PyArray_Check((PyObject*)candidate)) can_refer = false;

  if (can_refer && req_dtype &&
      !PyArray_EquivTypes(candidate->descr, req_dtype))
    can_refer = false;

  if (can_refer && !PyArray_ISCARRAY_RO(candidate)) can_refer = false;

  if (can_refer) {
    PyObject* tmp = PyArray_FromArray(candidate, 0, 0);
    return tp::make_non_null_object(tmp);
  }

  //copy
  TDEBUG1("[non-optimal] copying array-like object - cannot refer");
  PyObject* _ptr = (PyObject*)candidate;
#if NPY_FEATURE_VERSION > NUMPY16_API /* NumPy C-API version > 1.6 */
  int flags = NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ENSURECOPY|NPY_ARRAY_ENSUREARRAY;
#else
  int flags = NPY_C_CONTIGUOUS|NPY_ENSURECOPY|NPY_ENSUREARRAY;
#endif
  PyObject* tmp = PyArray_FromAny(_ptr, req_dtype, 0, 0, flags, 0);
  return tp::make_non_null_object(tmp);

}

static void derefer_ndarray (PyArrayObject* array) {
  Py_XDECREF(array);
}

static boost::shared_ptr<void> shared_from_ndarray (bp::object& o) {
  boost::shared_ptr<PyArrayObject> cache(TP_ARRAY(o), 
      std::ptr_fun(derefer_ndarray));
  Py_XINCREF(TP_OBJECT(o)); ///< makes sure it outlives this scope!
  return cache; //casts to b::shared_ptr<void>
}

tp::py_array::py_array(bp::object o, bp::object _dtype):
  m_is_numpy(true)
{
  if (TPY_ISNONE(o)) PYTHON_ERROR(TypeError, "You cannot pass 'None' as input parameter to C++-bound Torch methods that expect NumPy ndarrays (or blitz::Array<T,N>'s). Double-check your input!");

  bp::object mine = try_refer_ndarray(o, _dtype);

  //captures data from a numeric::array
  typeinfo_ndarray_(mine, m_type);

  //transforms the from boost::python ref counting to boost::shared_ptr<void>
  m_data = shared_from_ndarray(mine);

  //set-up the C-style pointer to this data
  m_ptr = static_cast<void*>(TP_ARRAY(mine)->data);
}

tp::py_array::py_array(const ca::interface& other) {
  set(other);
}

tp::py_array::py_array(boost::shared_ptr<ca::interface> other) {
  set(other);
}

tp::py_array::py_array(const ca::typeinfo& info) {
  set(info);
}

tp::py_array::~py_array() {
}

/**
 * Wrap a C-style pointer with a PyArrayObject
 */
static bp::object wrap_data (void* data, const ca::typeinfo& ti) {
  npy_intp shape[NPY_MAXDIMS];
  npy_intp stride[NPY_MAXDIMS];
  for (size_t k=0; k<ti.nd; ++k) {
    shape[k] = ti.shape[k];
    stride[k] = ti.item_size()*ti.stride[k];
  }
  PyObject* tmp = PyArray_New(&PyArray_Type, ti.nd,
        &shape[0], tp::type_to_num(ti.dtype), &stride[0], data, 0, 0, 0);
  return tp::make_non_null_object(tmp);
}

/**
 * New wrapper of the array
 */
static bp::object wrap_ndarray (const bp::object& a) {
  PyObject* tmp = PyArray_FromArray(TP_ARRAY(a), 0, 0); 
  return tp::make_non_null_object(tmp);
}

/**
 * Creates a new array from specifications
 */
static bp::object make_ndarray(int nd, npy_intp* dims, int type) {
  PyObject* tmp = PyArray_SimpleNew(nd, dims, type);
  return tp::make_non_null_object(tmp);
}

/**
 * New copy of the array from another array
 */
static bp::object copy_array (const bp::object& array) {
  PyArrayObject* _p = TP_ARRAY(array);
  bp::object retval = make_ndarray(_p->nd, _p->dimensions, _p->descr->type_num);
  PyArray_CopyInto(TP_ARRAY(retval), TP_ARRAY(array));
  return retval;
}

/**
 * Copies a data pointer and type into a new numpy array.
 */
static bp::object copy_data (const void* data, const ca::typeinfo& ti) {
  bp::object wrapped = wrap_data(const_cast<void*>(data), ti);
  bp::object retval = copy_array (wrapped);
  return retval;
}

void tp::py_array::set(const ca::interface& other) {
  TDEBUG1("[non-optimal] buffer copying operation being performed for " 
      << other.type().str());

  //performs a copy of the data into a numpy array
  bp::object mine = copy_data(other.ptr(), m_type);

  //captures data from a numeric::array
  typeinfo_ndarray_(mine, m_type);

  //transforms the from boost::python ref counting to boost::shared_ptr<void>
  m_data = shared_from_ndarray(mine);

  //set-up the C-style pointer to this data
  m_ptr = static_cast<void*>(TP_ARRAY(mine)->data);

  m_is_numpy = true;
}

void tp::py_array::set(boost::shared_ptr<ca::interface> other) {
  m_type = other->type();
  m_is_numpy = false;
  m_ptr = other->ptr();
  m_data = other->owner();
}

/**
 * Creates a new numpy array from a Torch::io::typeinfo object.
 */
static bp::object new_from_type (const ca::typeinfo& ti) {
  npy_intp shape[NPY_MAXDIMS];
  npy_intp stride[NPY_MAXDIMS];
  for (size_t k=0; k<ti.nd; ++k) {
    shape[k] = ti.shape[k];
    stride[k] = ti.item_size()*ti.stride[k];
  }
  PyObject* tmp = PyArray_New(&PyArray_Type, ti.nd, &shape[0], 
      tp::type_to_num(ti.dtype), &stride[0], 0, 0, 0, 0);
  return tp::make_non_null_object(tmp);
}

void tp::py_array::set (const ca::typeinfo& req) {
  if (m_type.is_compatible(req)) return; ///< nothing to do!
  
  TDEBUG1("[non-optimal?] buffer re-size being performed from " << m_type.str()
      << " to " << req.str());

  bp::object mine = new_from_type(req);

  //captures data from a numeric::array
  typeinfo_ndarray_(mine, m_type);

  //transforms the from boost::python ref counting to boost::shared_ptr<void>
  m_data = shared_from_ndarray(mine);

  //set-up the C-style pointer to this data
  m_ptr = static_cast<void*>(TP_ARRAY(mine)->data);

  m_is_numpy = true;
}

bp::object tp::py_array::copy(const bp::object& dtype) {
  return copy_data(m_ptr, m_type);
}

/**
 * Gets a read-only reference to a certain data. This recipe was originally
 * posted here:
 * http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
 *
 * But a better allocation strategy (that actually works) is posted here:
 * http://stackoverflow.com/questions/2924827/numpy-array-c-api
 */
static void DeleteSharedPointer (void* ptr) {
  typedef boost::shared_ptr<const void> type;
  delete static_cast<type*>(ptr);
}

static bp::object make_readonly (const void* data, const ca::typeinfo& ti,
    boost::shared_ptr<const void> owner) {

  bp::object retval = wrap_data(const_cast<void*>(data), ti);

  //resets the "WRITEABLE" flag
  TP_ARRAY(retval)->flags &= 
#if NPY_FEATURE_VERSION > NUMPY16_API /* NumPy C-API version > 1.6 */
    ~NPY_ARRAY_WRITEABLE
#else
    ~NPY_WRITEABLE
#endif
    ;

  //creates the shared pointer deallocator
  boost::shared_ptr<const void>* ptr = new boost::shared_ptr<const void>(owner);
  PyObject* py_sharedptr = PyCObject_FromVoidPtr(ptr, DeleteSharedPointer);

  if (!py_sharedptr) {
    PYTHON_ERROR(RuntimeError, "could not allocate space for deallocation object in read-only array::interface wrapping");
  }

  TP_ARRAY(retval)->base = py_sharedptr;

  return retval;
}

bp::object tp::py_array::pyobject() {
  if (m_is_numpy) {
    bp::object mine = tp::make_non_null_borrowed_object(boost::static_pointer_cast<PyObject>(m_data).get());
    return wrap_ndarray(mine);
  }

  //if you really want, I can wrap it for you, but in this case I'll make it
  //read-only and will associate the object deletion to my own data pointer.
  return make_readonly(m_ptr, m_type, m_data);
}

bool tp::py_array::is_writeable() const {
  return (!m_is_numpy || PyArray_ISWRITEABLE(boost::static_pointer_cast<PyArrayObject>(m_data).get()));
}

tp::ndarray::ndarray(bp::object array_like, bp::object dtype_like)
  : px(new tp::py_array(array_like, dtype_like)) { 
}

tp::ndarray::ndarray(bp::object array_like)
  : px(new tp::py_array(array_like, bp::object())) { 
  }

tp::ndarray::ndarray(const ca::typeinfo& info)
  : px(new tp::py_array(info)) { 
  }

tp::ndarray::~ndarray() { }

const ca::typeinfo& tp::ndarray::type() const {
  return px->type();
}

bp::object tp::ndarray::self() { return px->pyobject(); }

tp::const_ndarray::const_ndarray(bp::object array_like)
  : tp::ndarray(array_like) { 
  }

tp::const_ndarray::~const_ndarray() { }

