/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 27 Oct 08:15:25 2011
 *
 * @brief Implementation of NumPy array <=> Torch::io::buffer interface
 */

#include "io/python/npyarray.h"

namespace tp = Torch::python;
namespace io = Torch::io;
namespace bp = boost::python;
namespace tca = Torch::core::array;

static void typeinfo_ndarray (PyArrayObject* npy, io::typeinfo& i) {
  i.set<npy_intp>(tp::num_to_eltype(npy->descr->type_num), npy->nd, 
      npy->dimensions, npy->strides);
}

static void derefer_ndarray (PyArrayObject* array) {
  Py_DECREF(array);
}

/**
 * Creates a new numpy array from a Torch::io::typeinfo object.
 */
static PyArrayObject* new_from_type (const io::typeinfo& ti) {
  npy_intp shape[TORCH_MAX_DIM];
  npy_intp stride[TORCH_MAX_DIM];
  for (size_t k=0; k<ti.nd; ++k) {
    shape[k] = ti.shape[k];
    stride[k] = tca::getElementSize(ti.dtype)*ti.stride[k];
  }
  return reinterpret_cast<PyArrayObject*>(PyArray_New(&PyArray_Type, ti.nd,
        &shape[0], tp::eltype_to_num(ti.dtype), &stride[0], 0, 0, 0, 0));
}

/**
 * Wrap a C-style pointer with a PyArrayObject - you must derefer it when done.
 */
static PyArrayObject* wrap_data (void* data, const io::typeinfo& ti) {
  npy_intp shape[TORCH_MAX_DIM];
  npy_intp stride[TORCH_MAX_DIM];
  for (size_t k=0; k<ti.nd; ++k) {
    shape[k] = ti.shape[k];
    stride[k] = tca::getElementSize(ti.dtype)*ti.stride[k];
  }
  return reinterpret_cast<PyArrayObject*>(PyArray_New(&PyArray_Type, ti.nd,
        &shape[0], tp::eltype_to_num(ti.dtype), &stride[0], data, 0, 0, 0));
}

/**
 * New wrapper of the array
 */
static PyArrayObject* wrap_array (PyArrayObject* from, PyArray_Descr* dtype) {
  return reinterpret_cast<PyArrayObject*>(PyArray_FromArray(from, dtype, 0));
}

/**
 * New copy of the array from another array
 */
static PyArrayObject* copy_array (PyArrayObject* from) {
  PyArrayObject* retval = tp::make_ndarray(from->nd, from->dimensions,
      from->descr->type_num);
  PyArray_CopyInto(retval, from);
  return retval;
}

/**
 * Copies a data pointer and type into a new numpy array.
 */
static PyArrayObject* copy_data (const void* data, const io::typeinfo& ti) {
  PyArrayObject* wrapped = wrap_data(const_cast<void*>(data), ti);
  PyArrayObject* retval = copy_array (wrapped);
  derefer_ndarray(wrapped);
  return retval;
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

static PyArrayObject* make_readonly (const void* data, const io::typeinfo& ti,
    boost::shared_ptr<const void> owner) {

  PyArrayObject* retval = wrap_data(const_cast<void*>(data), ti);

  //resets the "WRITEABLE" flag
  retval->flags &= 
#if C_API_VERSION >= 6 /* NumPy C-API version > 1.6 */
    ~NPY_ARRAY_WRITEABLE
#else
    ~NPY_WRITEABLE
#endif
    ;

  if (!retval) 
    PYTHON_ERROR(RuntimeError, "could not allocate space for read-only array wrapper");

  //creates the shared pointer deallocator
  boost::shared_ptr<const void>* ptr = new boost::shared_ptr<const void>(owner);
  PyObject* py_sharedptr = PyCObject_FromVoidPtr(ptr, DeleteSharedPointer);

  if (!py_sharedptr) {
    Py_XDECREF(retval);
    PYTHON_ERROR(RuntimeError, "could not allocate space for deallocation object in read-only io::buffer wrapping");
  }

  retval->base = py_sharedptr;

  return retval;
}

PyArrayObject* tp::buffer_array (boost::shared_ptr<io::buffer> b) {
  boost::shared_ptr<tp::npyarray> npy = 
    boost::dynamic_pointer_cast<tp::npyarray>(b);
  if (npy) return npy->shallow_copy_force();
  return tp::buffer_array(*b);
}

PyArrayObject* tp::buffer_array (const io::buffer& b) {
  return make_readonly(b.ptr(), b.type(), b.owner());
}

tp::npyarray::npyarray(PyArrayObject* npy):
  m_is_numpy(true)
{
  //captures data from a numeric::array
  typeinfo_ndarray(npy, m_type);

  //can only deal with behaved C arrays with native byte-order
  assert_ndarray_byteorder(npy);
  assert_ndarray_behaved(npy);

  //now we capture a pointer to the passed array
  PyArrayObject* copy = wrap_array(npy, 0);
  boost::shared_ptr<PyArrayObject> cache(copy, std::ptr_fun(derefer_ndarray));
  m_data = cache;

  //set-up the C-style pointer to this data
  m_ptr = static_cast<void*>(npy->data);
}

tp::npyarray::npyarray(bp::object o, bp::object dtype):
  m_is_numpy(true)
{
  PyObject* ptr = o.ptr();
  PyArrayObject* copy = 0;

  PyArray_Descr* npy_dtype = 0;
  PyArray_DescrConverter2(dtype.ptr(), &npy_dtype);

  if (PyArray_Check(ptr)) {

    PyArrayObject* npy = (PyArrayObject*)ptr;

    //can only deal with behaved C arrays with native byte-order
    assert_ndarray_byteorder(npy);
    assert_ndarray_behaved(npy);

    copy = wrap_array(npy, npy_dtype);

  }

  else {

    //see if we can convert it -- this will copy the data, be warned.
    copy = (PyArrayObject*)PyArray_FromAny(ptr, npy_dtype, 0, 0, 0, 0);
    if (!copy) PYTHON_ERROR(TypeError, "object is not array-like or can be converted into an ndarray");

  }

  //captures data from a numeric::array
  typeinfo_ndarray(copy, m_type);

  //now we capture a pointer to the passed array
  boost::shared_ptr<PyArrayObject> cache(copy, std::ptr_fun(derefer_ndarray));
  m_data = cache;

  //set-up the C-style pointer to this data
  m_ptr = static_cast<void*>(copy->data);
}

tp::npyarray::npyarray(const io::buffer& other) {
  set(other);
}

tp::npyarray::npyarray(boost::shared_ptr<io::buffer> other) {
  set(other);
}

tp::npyarray::npyarray(const io::typeinfo& info) {
  set(info);
}

tp::npyarray::~npyarray() {
}

void tp::npyarray::set(const io::buffer& other) {
  m_type = other.type();

  //performs a copy of the data into a numpy array
  PyArrayObject* copy = copy_data(other.ptr(), m_type);
  boost::shared_ptr<PyArrayObject> cache(copy, std::ptr_fun(derefer_ndarray));

  m_ptr = copy->data;
  m_data = cache;
  
  m_is_numpy = true;
}

void tp::npyarray::set(boost::shared_ptr<io::buffer> other) {
  m_type = other->type();
  m_is_numpy = false;
  m_ptr = other->ptr();
  m_data = other->owner();
}

void tp::npyarray::set (const io::typeinfo& req) {
  m_type = req;
  
  PyArrayObject* arr = new_from_type(req);
  if (!arr) PYTHON_ERROR(RuntimeError, "cannot re-allocate buffer");

  boost::shared_ptr<PyArrayObject> cache(arr, std::ptr_fun(derefer_ndarray));
  
  m_ptr = arr->data;
  m_data = cache;

  m_is_numpy = true;
}

PyArrayObject* tp::npyarray::shallow_copy () {
  if (m_is_numpy)
    return wrap_array(boost::static_pointer_cast<PyArrayObject>(m_data).get(), 0);

  return 0;
}

PyArrayObject* tp::npyarray::deep_copy () {
  return copy_data(m_ptr, m_type);
}

PyArrayObject* tp::npyarray::shallow_copy_force () {
  if (m_is_numpy)
    return wrap_array(boost::static_pointer_cast<PyArrayObject>(m_data).get(), 0);

  //if you really want, I can wrap it for you, but in this case I'll make it
  //read-only and will associate the object deletion to my own data pointer.
  return make_readonly(m_ptr, m_type, m_data);
}
