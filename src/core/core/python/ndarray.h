/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief Automatic converters for numpy.ndarray.
 */

#ifndef TORCH_PYTHON_NDARRAY_H
#define TORCH_PYTHON_NDARRAY_H

#include <boost/python.hpp>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <blitz/array.h>

namespace boost { namespace python {

  /**
   * This class implements transparent binding between numpy ndarrays, to and
   * from C++. You can pass an ndarray in python and this will be what it looks
   * like from the C++ side. Automatic converters are registered into
   * boost::python so you don't have to do anything.
   */
  class ndarray : public boost::python::object {

    private:
      boost::python::object m_obj;

    public:

      /**
       * forwards all standard constructors into this class
       */
      BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(ndarray, object);

      /**
       * builds from an normal object
       */
      ndarray(const object& obj = object());

      /**
       * copy constructor
       */
      ndarray(const ndarray& obj);

      /**
       * builds from a particular blitz::Array<> type
       */
      template <typename T, int N> ndarray(const blitz::Array<T,N>& bz) {
        PyErr_SetString(PyExc_TypeError, "unsupported blitz::Array<T,N>");
        boost::python::throw_error_already_set();
      }

      /**
       * virtual d'tor
       */
      virtual ~ndarray();

      /**
       * points to another array
       */
      ndarray& operator=(const ndarray& other);

      /**
       * cheks if a given object does conform to the PyArrayObject API
       */
      void check_obj(const boost::python::object& obj) const;
    
      /**
       * returns the base object
       */
      const boost::python::object& get_obj() const;

      /**
       * Returns the number of dimensions for the given array
       */
      Py_ssize_t ndim() const;

      /**
       * Returns the total number of elements for this array.
       */
      Py_ssize_t size() const;

      /**
       * Returns a pointer to the array shape
       */
      const npy_intp* shape() const;

      /**
       * Returns a pointer to the array strides (C storage)
       */
      const npy_intp* strides() const;

      /**
       * Returns the size of each element in this array.
       */
      Py_ssize_t itemsize() const;

      /**
       * Returns a pointer to the C data storage
       */
      const void* data() const;

      /**
       * Returns a clone of this array, enforcing a new type.
       */
      ndarray astype(NPY_TYPES cast_to) const;

      /**
       * Returns the internal element type of this array.
       */
      NPY_TYPES dtype() const;

      /**
       * Converts a Numpy array to a blitz one copying the data and casting to
       * the destination element type. Please note that this is just a place
       * holder, only the full specializations are actual valid
       * implementations.
       */
      template<typename T, int N>
        boost::shared_ptr<blitz::Array<T,N> > to_blitz () const {
          PyErr_SetString(PyExc_TypeError, "unsupported blitz::Array<T,N>");
          boost::python::throw_error_already_set();
        }

  };

  ndarray new_ndarray(int len, npy_intp* shape, NPY_TYPES dtype);

# define NDARRAY_SPECIALIZATIONS(BZ_ELEMENT_TYPE) \
  template<> ndarray::ndarray(const blitz::Array<BZ_ELEMENT_TYPE,1>& bz); \
  template<> ndarray::ndarray(const blitz::Array<BZ_ELEMENT_TYPE,2>& bz); \
  template<> ndarray::ndarray(const blitz::Array<BZ_ELEMENT_TYPE,3>& bz); \
  template<> ndarray::ndarray(const blitz::Array<BZ_ELEMENT_TYPE,4>& bz); \
  template<> boost::shared_ptr<blitz::Array<BZ_ELEMENT_TYPE,1> > ndarray::to_blitz<BZ_ELEMENT_TYPE,1>() const; \
  template<> boost::shared_ptr<blitz::Array<BZ_ELEMENT_TYPE,2> > ndarray::to_blitz<BZ_ELEMENT_TYPE,2>() const; \
  template<> boost::shared_ptr<blitz::Array<BZ_ELEMENT_TYPE,3> > ndarray::to_blitz<BZ_ELEMENT_TYPE,3>() const; \
  template<> boost::shared_ptr<blitz::Array<BZ_ELEMENT_TYPE,4> > ndarray::to_blitz<BZ_ELEMENT_TYPE,4>() const 
  NDARRAY_SPECIALIZATIONS(bool);
  NDARRAY_SPECIALIZATIONS(int8_t);
  NDARRAY_SPECIALIZATIONS(uint8_t);
  NDARRAY_SPECIALIZATIONS(int16_t);
  NDARRAY_SPECIALIZATIONS(uint16_t);
  NDARRAY_SPECIALIZATIONS(int32_t);
  NDARRAY_SPECIALIZATIONS(uint32_t);
  NDARRAY_SPECIALIZATIONS(int64_t);
  NDARRAY_SPECIALIZATIONS(uint64_t);
  NDARRAY_SPECIALIZATIONS(float);
  NDARRAY_SPECIALIZATIONS(double);
  NDARRAY_SPECIALIZATIONS(long double);
  NDARRAY_SPECIALIZATIONS(std::complex<float>);
  NDARRAY_SPECIALIZATIONS(std::complex<double>);
  NDARRAY_SPECIALIZATIONS(std::complex<long double>);
# undef NDARRAY_SPECIALIZATIONS

}} // namespace boost::python

#endif //TORCH_PYTHON_NDARRAY_H
