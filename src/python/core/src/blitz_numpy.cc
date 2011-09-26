/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sat 24 Sep 05:01:54 2011 CEST
 *
 * @brief Automatic converters to-from python for blitz::Array's
 */

#include <boost/python.hpp>

namespace bp = boost::python;

/**
 * Objects of this type create a binding between blitz::Array<T,N> and
 * NumPy arrays. You can specify a NumPy array as a parameter to a
 * bound method that would normally receive a blitz::Array<T,N> or a const
 * blitz::Array<T,N>& and the conversion will just magically happen, as
 * efficiently as possible.
 *
 * Please note that passing by value should be avoided as much as possible. In
 * this mode, the underlying method will still be able to alter the underlying
 * array storage area w/o being able to modify the array itself, causing a
 * gigantic mess. If you want to make something close to pass-by-value, just
 * pass by non-const reference instead.
 */
template <typename T, int N>
struct bz_from_npy {

  typedef typename blitz::Array<T,N> array_type;

  /**
   * Registers converter from numpy array into a blitz::Array<T,N>
   */
  bz_from_npy() {
    bp::converter::registry::push_back(&convertible, &construct, 
        bp::type_id<array_type>());
  }

  /**
   * This method will determine if the input python object is convertible into
   * a TinyVector<T,N>
   *
   * Conditions:
   * - The input object has to be a numpy array
   */
  static void* convertible(PyObject* obj_ptr) {
    if (PyArray_Check(obj_ptr)) return obj_ptr;
    return 0;
  }

  /**
   * This method will finally construct the C++ element out of the python
   * object that was input. Please note that when boost::python reaches this
   * method, the object has already been checked for convertibility.
   */
  static void construct(PyObject* obj_ptr,
      bp::converter::rvalue_from_python_stage1_data* data) {

    //black-magic required to setup the blitz::Array<> storage area
    void* storage = ((boost::python::converter::rvalue_from_python_storage<array_type>*)data)->storage.bytes;
    new (storage) array_type();
    data->convertible = storage;
    array_type& result = *((array_type*)storage);

    //now we proceed to the conversion.
    boost::python::handle<> obj(obj_ptr);

    /**
     * We need to tackle the following use cases:
     *
     * 1. The NumPy array type T is the right type for the conversion, in which
     *    case we just take a pointer to the data and instantiate a
     *    blitz::Array<T,N> out of that
     * 2. The type T is wrong and we need to cast it to a new data type.
     */
     
};

/**
 * Objects of this type bind TinyVector<T,N> to python tuples. Your method
 * generates as output an object of this type and the object will be
 * automatically converted into a python tuple.
 */
template <typename T, int N>
struct tinyvec_to_tuple {
  typedef typename blitz::TinyVector<T,N> array_type;

  static PyObject* convert(const array_type& tv) {
    boost::python::list result;
    typedef typename array_type::const_iterator const_iter;
    for(const_iter p=tv.begin();p!=tv.end();++p) {
      result.append(boost::python::object(*p));
    }
    return boost::python::incref(boost::python::tuple(result).ptr());
  }

  static const PyTypeObject* get_pytype() { return &PyTuple_Type; }

};

template <typename T, int N>
void register_tinyvec_to_tuple() {
  bp::to_python_converter<typename blitz::TinyVector<T,N>, 
                          tinyvec_to_tuple<T,N>
#if defined BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
                          ,true
#endif
              >();
}

void bind_core_array_tinyvector () {

  /**
   * The following struct constructors will make sure we can input
   * blitz::TinyVector<T,N> in our bound C++ routines w/o needing to specify
   * special converters each time. The rvalue converters allow boost::python to
   * automatically map the following inputs:
   *
   * a) const blitz::TinyVector<T,N>& (pass by const reference)
   * b) blitz::TinyVector<T,N> (pass by value)
   *
   * Please note that the last case:
   * 
   * c) blitz::TinyVector<T,N>& (pass by non-const reference)
   *
   * is NOT covered by these converters. The reason being that because the
   * object may be changed, there is no way for boost::python to update the
   * original python object, in a sensible manner, at the return of the method.
   *
   * Avoid passing by non-const reference in your methods.
   */
  tinyvec_from_sequence<int,1>();
  tinyvec_from_sequence<int,2>();
  tinyvec_from_sequence<int,3>();
  tinyvec_from_sequence<int,4>();
  tinyvec_from_sequence<int,5>();
  tinyvec_from_sequence<int,6>();
  tinyvec_from_sequence<int,7>();
  tinyvec_from_sequence<int,8>();
  tinyvec_from_sequence<int,9>();
  tinyvec_from_sequence<int,10>();
  tinyvec_from_sequence<int,11>();
  tinyvec_from_sequence<uint64_t,1>();
  tinyvec_from_sequence<uint64_t,2>();
  tinyvec_from_sequence<uint64_t,3>();
  tinyvec_from_sequence<uint64_t,4>();
  tinyvec_from_sequence<uint64_t,5>();
  tinyvec_from_sequence<uint64_t,6>();
  tinyvec_from_sequence<uint64_t,7>();
  tinyvec_from_sequence<uint64_t,8>();
  tinyvec_from_sequence<uint64_t,9>();
  tinyvec_from_sequence<uint64_t,10>();
  tinyvec_from_sequence<uint64_t,11>();
# if defined(HAVE_BLITZ_SPECIAL_TYPES)
  if (typeid(int) != typeid(blitz::diffType)) {
    tinyvec_from_sequence<blitz::diffType,1>();
    tinyvec_from_sequence<blitz::diffType,2>();
    tinyvec_from_sequence<blitz::diffType,3>();
    tinyvec_from_sequence<blitz::diffType,4>();
    tinyvec_from_sequence<blitz::diffType,5>();
    tinyvec_from_sequence<blitz::diffType,6>();
    tinyvec_from_sequence<blitz::diffType,7>();
    tinyvec_from_sequence<blitz::diffType,8>();
    tinyvec_from_sequence<blitz::diffType,9>();
    tinyvec_from_sequence<blitz::diffType,10>();
    tinyvec_from_sequence<blitz::diffType,11>();
  }
# endif //defined(HAVE_BLITZ_SPECIAL_TYPES)

  /**
   * The following struct constructors will make C++ return values of type
   * blitz::TinyVector<T,N> to show up in the python side as tuples.
   */
  register_tinyvec_to_tuple<int,1>();
  register_tinyvec_to_tuple<int,2>();
  register_tinyvec_to_tuple<int,3>();
  register_tinyvec_to_tuple<int,4>();
  register_tinyvec_to_tuple<int,5>();
  register_tinyvec_to_tuple<int,6>();
  register_tinyvec_to_tuple<int,7>();
  register_tinyvec_to_tuple<int,8>();
  register_tinyvec_to_tuple<int,9>();
  register_tinyvec_to_tuple<int,10>();
  register_tinyvec_to_tuple<int,11>();
  register_tinyvec_to_tuple<uint64_t,1>();
  register_tinyvec_to_tuple<uint64_t,2>();
  register_tinyvec_to_tuple<uint64_t,3>();
  register_tinyvec_to_tuple<uint64_t,4>();
  register_tinyvec_to_tuple<uint64_t,5>();
  register_tinyvec_to_tuple<uint64_t,6>();
  register_tinyvec_to_tuple<uint64_t,7>();
  register_tinyvec_to_tuple<uint64_t,8>();
  register_tinyvec_to_tuple<uint64_t,9>();
  register_tinyvec_to_tuple<uint64_t,10>();
  register_tinyvec_to_tuple<uint64_t,11>();
# if defined(HAVE_BLITZ_SPECIAL_TYPES)
  if (typeid(int) != typeid(blitz::diffType)) {
    register_tinyvec_to_tuple<blitz::diffType,1>();
    register_tinyvec_to_tuple<blitz::diffType,2>();
    register_tinyvec_to_tuple<blitz::diffType,3>();
    register_tinyvec_to_tuple<blitz::diffType,4>();
    register_tinyvec_to_tuple<blitz::diffType,5>();
    register_tinyvec_to_tuple<blitz::diffType,6>();
    register_tinyvec_to_tuple<blitz::diffType,7>();
    register_tinyvec_to_tuple<blitz::diffType,8>();
    register_tinyvec_to_tuple<blitz::diffType,9>();
    register_tinyvec_to_tuple<blitz::diffType,10>();
    register_tinyvec_to_tuple<blitz::diffType,11>();
  }
# endif //defined(HAVE_BLITZ_SPECIAL_TYPES)

}
