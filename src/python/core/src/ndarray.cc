/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  1 Nov 11:24:40 2011
 *
 * @brief Implementation of the ndarray class
 */

#include <boost/python/numeric.hpp>
#include <stdexcept>
#include <dlfcn.h>

#define torch_IMPORT_ARRAY
#include "core/python/ndarray.h"
#undef torch_IMPORT_ARRAY

#include "core/logging.h"

namespace bp = boost::python;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

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
}

/***************************************************************************
 * Dtype (PyArray_Descr) manipulations                                     *
 ***************************************************************************/

/**
 * Creates an auto-deletable bp::object out of a standard Python object.
 */
bp::object make_non_null_object(PyObject* obj) {
  bp::handle<> hdl(obj); //< raises if NULL
  bp::object retval(hdl);
  return retval;
}

/**
 * Creates an auto-deletable bp::object out of a standard Python object.
 */
bp::object make_maybe_null_object(PyObject* obj) {
  bp::handle<> hdl(bp::allow_null(obj));
  bp::object retval(hdl);
  return retval;
}

/**
 * Creates an auto-deletable bp::object out of a standard Python object.
 */
bp::object make_non_null_borrowed_object(PyObject* obj) {
  bp::handle<> hdl(bp::borrowed(obj));
  bp::object retval(hdl);
  return retval;
}

int tp::type_to_num(ca::ElementType type) {

  switch(type) {
    case ca::t_bool:
      return NPY_BOOL;
    case ca::t_int8:
      return NPY_BYTE;
    case ca::t_int16:
      return NPY_SHORT;
    case ca::t_int32:
      return NPY_INT;
    case ca::t_int64:
      return (sizeof(long) == 8)?NPY_LONG:NPY_LONGLONG;
    case ca::t_uint8:
      return NPY_UBYTE;
    case ca::t_uint16:
      return NPY_USHORT;
    case ca::t_uint32:
      return NPY_UINT;
    case ca::t_uint64:
      return (sizeof(unsigned long) == 8)?NPY_ULONG:NPY_ULONGLONG;
    case ca::t_float32:
      return NPY_FLOAT;
    case ca::t_float64:
      return NPY_DOUBLE;
    case ca::t_float128:
      return NPY_LONGDOUBLE;
    case ca::t_complex64:
      return NPY_CFLOAT;
    case ca::t_complex128:
      return NPY_CDOUBLE;
    case ca::t_complex256:
      return NPY_CLONGDOUBLE;
    default:
      PYTHON_ERROR(TypeError, "unsupported C++ element type -- debug me!");
  }

}

ca::ElementType tp::num_to_type(int num) {

  switch(num) {
    case NPY_BOOL:
      return ca::t_uint8;
    case NPY_BYTE:
      return ca::t_int8;
    case NPY_SHORT:
      return ca::t_int16;
    case NPY_INT:
      return ca::t_int32;
    case NPY_LONG:
      return (sizeof(long) == 8)?ca::t_int64:ca::t_int32;
    case NPY_LONGLONG:
      return ca::t_int64;
    case NPY_UBYTE:
      return ca::t_uint8;
    case NPY_USHORT:
      return ca::t_uint16;
    case NPY_UINT:
      return ca::t_uint32;
    case NPY_ULONG:
      return (sizeof(unsigned long) == 8)?ca::t_uint64:ca::t_uint32;
    case NPY_ULONGLONG:
      return ca::t_uint64;
    case NPY_FLOAT:
      return ca::t_float32;
    case NPY_DOUBLE:
      return ca::t_float64;
    case NPY_LONGDOUBLE:
      return ca::t_float128;
    case NPY_CFLOAT:
      return ca::t_complex64;
    case NPY_CDOUBLE:
      return ca::t_complex128;
    case NPY_CLONGDOUBLE:
      return ca::t_complex256;
    default:
      PYTHON_ERROR(TypeError, "unsupported NumPy element type -- debug me!");
  }

}

template <> int tp::ctype_to_num<bool>(void) 
{ return NPY_BOOL; }
template <> int tp::ctype_to_num<signed char>(void) 
{ return NPY_BYTE; }
template <> int tp::ctype_to_num<unsigned char>(void) 
{ return NPY_UBYTE; }
template <> int tp::ctype_to_num<short>(void) 
{ return NPY_SHORT; }
template <> int tp::ctype_to_num<unsigned short>(void) 
{ return NPY_USHORT; }
template <> int tp::ctype_to_num<int>(void) 
{ return NPY_INT; }
template <> int tp::ctype_to_num<unsigned int>(void) 
{ return NPY_UINT; }
template <> int tp::ctype_to_num<long>(void)
{ return NPY_LONG; }
template <> int tp::ctype_to_num<unsigned long>(void)
{ return NPY_ULONG; }
template <> int tp::ctype_to_num<long long>(void)
{ return NPY_LONGLONG; }
template <> int tp::ctype_to_num<unsigned long long>(void)
{ return NPY_ULONGLONG; }
template <> int tp::ctype_to_num<float>(void)
{ return NPY_FLOAT; }
template <> int tp::ctype_to_num<double>(void) 
{ return NPY_DOUBLE; }
template <> int tp::ctype_to_num<long double>(void) 
{ return NPY_LONGDOUBLE; }
template <> int tp::ctype_to_num<std::complex<float> >(void)
{ return NPY_CFLOAT; }
template <> int tp::ctype_to_num<std::complex<double> >(void) 
{ return NPY_CDOUBLE; }
template <> int tp::ctype_to_num<std::complex<long double> >(void) 
{ return NPY_CLONGDOUBLE; }

#define TP_DESCR(x) ((PyArray_Descr*)x.ptr())

tp::dtype::dtype (bp::object dtype_like) {
  PyArray_Descr* tmp = 0;
  PyArray_DescrConverter2(dtype_like.ptr(), &tmp); //returns new ref
  m_self = make_non_null_object((PyObject*)tmp);
}

tp::dtype::dtype(ca::ElementType eltype) {
  PyArray_Descr* tmp = PyArray_DescrFromType(tp::type_to_num(eltype)); //new ref
  m_self = make_non_null_object((PyObject*)tmp);
}

tp::dtype::~dtype() { }

bool tp::dtype::has_native_byteorder() const {
  return (PyArray_EquivByteorders(TP_DESCR(m_self)->byteorder, NPY_NATIVE) || 
      TP_DESCR(m_self)->elsize == 1);
}

bool tp::dtype::has_type(ca::ElementType _eltype) const {
  return eltype() == _eltype;
}

ca::ElementType tp::dtype::eltype() const {
  return tp::num_to_type(TP_DESCR(m_self)->type_num);
}

/***************************************************************************
 * Ndarray (PyArrayObject) manipulations                                   *
 ***************************************************************************/

#define TP_ARRAY(x) ((PyArrayObject*)x.ptr())
#define TP_OBJECT(x) (x.ptr())

/**
 * Returns either a reference or a copy to the given array_like object,
 * depending on the following requirements for referral:
 *
 * 0. The pointed object is a numpy.ndarray
 * 1. The array type description type_num matches
 * 2. The array is C-style, contiguous and aligned
 */
static bp::object try_refer_ndarray (boost::python::object array_like, 
    boost::python::object dtype) {
  
  PyArrayObject* candidate = TP_ARRAY(array_like);

  bool can_refer = true;

  if (!PyArray_Check((PyObject*)candidate)) can_refer = false; //check 0
  
  if (!(PyArray_EquivByteorders(candidate->descr->byteorder, NPY_NATIVE) ||
        candidate->descr->elsize == 1)) can_refer = false; //check 1.a
        
  if (!dtype.is_none() && 
      candidate->descr->type_num != TP_DESCR(dtype)->type_num)
    can_refer = false; //check 1.b
 
  //tests the following: NPY_ARRAY_C_CONTIGUOUS and NPY_ARRAY_ALIGNED
  if (!PyArray_ISCARRAY_RO(candidate)) can_refer = false; //check 2

  PyObject* tmp = 0;

  if (can_refer) { //wrap
    tmp = PyArray_FromArray(candidate, 0, 0);
  }

  else { //copy
    TDEBUG1("[non-optimal] copying array-like object - cannot refer");
    PyObject* _ptr = (PyObject*)candidate;
    if (dtype.is_none())  tmp = PyArray_FromAny(_ptr, 0, 0, 0, 0, 0);
    else tmp = PyArray_FromAny(_ptr, TP_DESCR(dtype), 0, 0, 0, 0);
  }

  return make_non_null_object(tmp);
}

static void typeinfo_ndarray (const bp::object& o, ca::typeinfo& i) {
  PyArrayObject* npy = TP_ARRAY(o);
  i.set<npy_intp>(tp::num_to_type(npy->descr->type_num), npy->nd, 
      npy->dimensions, npy->strides);
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

tp::ndarray::ndarray(bp::object o, bp::object _dtype):
  m_is_numpy(true),
  m_dtype(_dtype)
{
  bp::object mine = try_refer_ndarray(o, m_dtype);

  //captures data from a numeric::array
  typeinfo_ndarray(mine, m_type);

  //transforms the from boost::python ref counting to boost::shared_ptr<void>
  m_data = shared_from_ndarray(mine);

  //set-up the C-style pointer to this data
  m_ptr = static_cast<void*>(TP_ARRAY(mine)->data);
}

tp::ndarray::ndarray(const ca::interface& other) {
  set(other);
}

tp::ndarray::ndarray(boost::shared_ptr<ca::interface>& other) {
  set(other);
}

tp::ndarray::ndarray(const ca::typeinfo& info) {
  set(info);
}

tp::ndarray::~ndarray() {
}

/**
 * Wrap a C-style pointer with a PyArrayObject
 */
static bp::object wrap_data (void* data, const ca::typeinfo& ti) {
  npy_intp shape[TORCH_MAX_DIM];
  npy_intp stride[TORCH_MAX_DIM];
  for (size_t k=0; k<ti.nd; ++k) {
    shape[k] = ti.shape[k];
    stride[k] = ti.item_size()*ti.stride[k];
  }
  PyObject* tmp = PyArray_New(&PyArray_Type, ti.nd,
        &shape[0], tp::type_to_num(ti.dtype), &stride[0], data, 0, 0, 0);
  return make_non_null_object(tmp);
}

/**
 * New wrapper of the array
 */
static bp::object wrap_ndarray (const bp::object& a) {
  PyObject* tmp = PyArray_FromArray(TP_ARRAY(a), 0, 0); 
  return make_non_null_object(tmp);
}

/**
 * Creates a new array from specifications
 */
static bp::object make_ndarray(int nd, npy_intp* dims, int type) {
  PyObject* tmp = PyArray_SimpleNew(nd, dims, type);
  return make_non_null_object(tmp);
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

void tp::ndarray::set(const ca::interface& other) {
  TDEBUG1("[non-optimal] buffer copying operation being performed for " 
      << other.type().str());

  //performs a copy of the data into a numpy array
  bp::object mine = copy_data(other.ptr(), m_type);

  //captures data from a numeric::array
  typeinfo_ndarray(mine, m_type);

  //transforms the from boost::python ref counting to boost::shared_ptr<void>
  m_data = shared_from_ndarray(mine);

  //set-up the C-style pointer to this data
  m_ptr = static_cast<void*>(TP_ARRAY(mine)->data);

  m_is_numpy = true;
}

void tp::ndarray::set(boost::shared_ptr<ca::interface>& other) {
  m_type = other->type();
  m_is_numpy = false;
  m_ptr = other->ptr();
  m_data = other->owner();
}

/**
 * Creates a new numpy array from a Torch::io::typeinfo object.
 */
static bp::object new_from_type (const ca::typeinfo& ti) {
  npy_intp shape[TORCH_MAX_DIM];
  npy_intp stride[TORCH_MAX_DIM];
  for (size_t k=0; k<ti.nd; ++k) {
    shape[k] = ti.shape[k];
    stride[k] = ti.item_size()*ti.stride[k];
  }
  PyObject* tmp = PyArray_New(&PyArray_Type, ti.nd, &shape[0], 
      tp::type_to_num(ti.dtype), &stride[0], 0, 0, 0, 0);
  return make_non_null_object(tmp);
}

void tp::ndarray::set (const ca::typeinfo& req) {
  if (m_type.is_compatible(req)) return; ///< nothing to do!
  
  TDEBUG1("[non-optimal?] buffer re-size being performed from " << m_type.str()
      << " to " << req.str());

  bp::object mine = new_from_type(req);

  //captures data from a numeric::array
  typeinfo_ndarray(mine, m_type);

  //transforms the from boost::python ref counting to boost::shared_ptr<void>
  m_data = shared_from_ndarray(mine);

  //set-up the C-style pointer to this data
  m_ptr = static_cast<void*>(TP_ARRAY(mine)->data);

  m_is_numpy = true;
}

bp::object tp::ndarray::copy(const bp::object& dtype) {
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
#if C_API_VERSION >= 6 /* NumPy C-API version > 1.6 */
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

bp::object tp::ndarray::pyobject() {
  if (m_is_numpy) {
    bp::object mine = make_non_null_borrowed_object(boost::static_pointer_cast<PyObject>(m_data).get());
    return wrap_ndarray(mine);
  }

  //if you really want, I can wrap it for you, but in this case I'll make it
  //read-only and will associate the object deletion to my own data pointer.
  return make_readonly(m_ptr, m_type, m_data);
}

bool tp::ndarray::is_writeable() const {
  return (!m_is_numpy || PyArray_ISWRITEABLE(boost::static_pointer_cast<PyArrayObject>(m_data).get()));
}
