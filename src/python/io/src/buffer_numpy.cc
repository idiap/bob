/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 27 Oct 10:03:13 2011
 *
 * @brief Automatic io::buffer conversion to/from NumPy.
 */

#include "io/python/pyio.h"

namespace io = Torch::io;
namespace bp = boost::python;
namespace tp = Torch::python;

/**
 * Objects of this type create a binding between io::buffer and NumPy arrays.
 * You can specify a NumPy array as a parameter to a bound method that would
 * normally receive a io::buffer or a const io::buffer& and the conversion will
 * just magically happen, as efficiently as possible.
 */
struct buffer_from_npy {

  /**
   * Registers converter from numpy array into an io::buffer 
   */
  buffer_from_npy() {
    bp::converter::registry::push_back(&convertible, &construct, 
        bp::type_id<io::buffer>());
  }

  /**
   * This method will determine if the input python object is convertible into
   * an io::buffer
   */
  static void* convertible(PyObject* obj_ptr) {
    
    PyArrayObject *arr = NULL;
    PyArray_Descr *dtype = NULL;
    int ndim = 0;
    npy_intp dims[NPY_MAXDIMS];

    /**
     * double-checks object "obj_ptr" for convertibility using NumPy's
     * facilities. Returns non-zero if fails. 
     *
     * Two scenarios:
     *
     * 1) "arr" is filled, other parameters are untouched. Means the input
     * object is already an array of the said type. 
     *
     * 2) "arr" is not filled, but input object is convertible with the
     * required specifications.
     */
    if (tp::check_ndarray(obj_ptr, 0, 0, dtype, ndim, dims, arr) != 0) {
      return 0;
    }

    if (arr) { //check arr properties
      if (arr->nd <= TORCH_MAX_DIM
          && tp::check_ndarray_byteorder(arr->descr)
         ) {
        return obj_ptr;
      }
    }

    else { //it is not a native array, see if a cast would work...

      //"dry-run" cast to make sure we can make an array out of the object
      if (tp::check_ndarray(obj_ptr,0,0,dtype,ndim,dims,arr) != 0) {
        return 0;
      }

      //check dimensions and byteorder
      if (ndim <= TORCH_MAX_DIM
          && tp::check_ndarray_byteorder(dtype)
         ) {
        return obj_ptr;
      }

    }

    return 0; //cannot be converted, if you get to this point...
  }

  /**
   * This method will finally construct the C++ element out of the python
   * object that was input. Please note that when boost::python reaches this
   * method, the object has already been checked for convertibility.
   */
  static void construct(PyObject* obj_ptr,
      bp::converter::rvalue_from_python_stage1_data* data) {

    //black-magic required to setup the io::buffer storage area
    void* storage = ((boost::python::converter::rvalue_from_python_storage<tp::npyarray>*)data)->storage.bytes;

    //now we proceed to the conversion -- take a look at the scenarios above

    //conversion happens in the most efficient way possible.
    PyArrayObject *arr = NULL;
    PyArray_Descr *dtype = NULL;
    int ndim = 0;
    npy_intp dims[NPY_MAXDIMS];

    /**
     * double-checks object "obj_ptr" for convertibility using NumPy's
     * facilities. Returns non-zero if fails. 
     *
     * Two scenarios:
     *
     * 1) "arr" is filled, other parameters are untouched. Means the input
     * object is already an array of the said type. 
     *
     * 2) "arr" is not filled, but input object is convertible with the
     * required specifications.
     */
    if (tp::check_ndarray(obj_ptr, 0, 0, dtype, ndim, dims, arr) != 0) {
      //this should never happen as the object has already been checked
      PYTHON_ERROR(TypeError, "object cannot be converted to blitz::Array");
    }

    PyArrayObject* arrobj; ///< holds our ndarray

    if (arr) { //perfect match, just grab the memory area -- fastest option
      arrobj = reinterpret_cast<PyArrayObject*>(obj_ptr);
    }
    else { //needs copying
      arrobj = tp::copy_ndarray(obj_ptr, 0, ndim);
    }

    new (storage) tp::npyarray(arrobj); //place operator
    data->convertible = storage;

    //we should dereference ourselves if we created a new array -- remember
    //that the npyarray object will reference it internally by itself.
    if (!arr) Py_DECREF(arrobj);

  }

};

/**
 * Objects of this type bind io::buffers to numpy arrays. Your method
 * generates as output an object of this type and the object will be
 * automatically converted into a Numpy array.
 */
struct buffer_to_npy {

  static PyObject* convert(const io::buffer& tv) {
    return (PyObject*)tp::buffer_array(tv);
  }

  static const PyTypeObject* get_pytype() { return &PyArray_Type; }

};

void register_buffer_to_npy() {
  bp::to_python_converter<io::buffer, buffer_to_npy
#if defined BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
                          ,true
#endif
              >();
}

void bind_io_buffer_numpy () {
   buffer_from_npy();
   register_buffer_to_npy();
}
