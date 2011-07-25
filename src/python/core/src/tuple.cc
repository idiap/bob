/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 25 Jul 08:59:27 2011 CEST
 *
 * @brief Automatic converters to-from python for boost::tuple<T,T,...>
 */

#include <boost/python.hpp>
#include <boost/format.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/preprocessor.hpp>
#include <stdint.h>

using namespace boost::python;

//A helper to set the value of a tuple from a python iterator
template <typename TupleType, int N>
static int tset(TupleType& t, PyObject* obj_iter) {
  handle<> py_elem_hdl(allow_null(PyIter_Next(obj_iter)));
  if (PyErr_Occurred()) throw_error_already_set();
  if (!py_elem_hdl.get()) return 0; // end of iteration
  object py_elem_obj(py_elem_hdl);
  extract<typename TupleType::head_type> elem_proxy(py_elem_obj);
  boost::get<N>(t) = elem_proxy();
  return 1;
}

//TODO: Re-encode the TypeChooser using boost::mpl and avoid all this code
//repetition.

//A type helper to select the right tuple type for the bindings
template <typename T, int N> struct TypeChooser;

template <typename T> struct TypeChooser<T,1> {
  typedef typename boost::tuple<T> value_type;

  static int set(value_type& t, PyObject* obj_iter) {
    return tset<value_type,0>(t, obj_iter);
  }

  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
  }
};

template <typename T> struct TypeChooser<T,2> {
  typedef typename boost::tuple<T,T> value_type;
  
  static int set(value_type& t, PyObject* obj_iter) {
    int retval = 0;
    retval += tset<value_type,0>(t, obj_iter); if (retval != 1) return retval;
    retval += tset<value_type,1>(t, obj_iter);
    return retval;
  }

  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
    l.append(boost::get<1>(t));
  }
};

template <typename T> struct TypeChooser<T,3> {
  typedef typename boost::tuple<T,T,T> value_type;
  
  static int set(value_type& t, PyObject* obj_iter) {
    int retval = 0;
    retval += tset<value_type,0>(t, obj_iter); if (retval != 1) return retval;
    retval += tset<value_type,1>(t, obj_iter); if (retval != 2) return retval;
    retval += tset<value_type,2>(t, obj_iter);
    return retval;
  }
  
  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
    l.append(boost::get<1>(t));
    l.append(boost::get<2>(t));
  }
};

template <typename T> struct TypeChooser<T,4> {
  typedef typename boost::tuple<T,T,T,T> value_type;
  
  static int set(value_type& t, PyObject* obj_iter) {
    int retval = 0;
    retval += tset<value_type,0>(t, obj_iter); if (retval != 1) return retval;
    retval += tset<value_type,1>(t, obj_iter); if (retval != 2) return retval;
    retval += tset<value_type,2>(t, obj_iter); if (retval != 3) return retval;
    retval += tset<value_type,3>(t, obj_iter);
    return retval;
  }
  
  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
    l.append(boost::get<1>(t));
    l.append(boost::get<2>(t));
    l.append(boost::get<3>(t));
  }
};

template <typename T> struct TypeChooser<T,5> {
  typedef typename boost::tuple<T,T,T,T,T> value_type;
  
  static int set(value_type& t, PyObject* obj_iter) {
    int retval = 0;
    retval += tset<value_type,0>(t, obj_iter); if (retval != 1) return retval;
    retval += tset<value_type,1>(t, obj_iter); if (retval != 2) return retval;
    retval += tset<value_type,2>(t, obj_iter); if (retval != 3) return retval;
    retval += tset<value_type,3>(t, obj_iter); if (retval != 4) return retval;
    retval += tset<value_type,4>(t, obj_iter);
    return retval;
  }
  
  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
    l.append(boost::get<1>(t));
    l.append(boost::get<2>(t));
    l.append(boost::get<3>(t));
    l.append(boost::get<4>(t));
  }
};

template <typename T> struct TypeChooser<T,6> {
  typedef typename boost::tuple<T,T,T,T,T,T> value_type;
  
  static int set(value_type& t, PyObject* obj_iter) {
    int retval = 0;
    retval += tset<value_type,0>(t, obj_iter); if (retval != 1) return retval;
    retval += tset<value_type,1>(t, obj_iter); if (retval != 2) return retval;
    retval += tset<value_type,2>(t, obj_iter); if (retval != 3) return retval;
    retval += tset<value_type,3>(t, obj_iter); if (retval != 4) return retval;
    retval += tset<value_type,4>(t, obj_iter); if (retval != 5) return retval;
    retval += tset<value_type,5>(t, obj_iter);
    return retval;
  }
  
  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
    l.append(boost::get<1>(t));
    l.append(boost::get<2>(t));
    l.append(boost::get<3>(t));
    l.append(boost::get<4>(t));
    l.append(boost::get<5>(t));
  }
};

template <typename T> struct TypeChooser<T,7> {
  typedef typename boost::tuple<T,T,T,T,T,T,T> value_type;
  
  static int set(value_type& t, PyObject* obj_iter) {
    int retval = 0;
    retval += tset<value_type,0>(t, obj_iter); if (retval != 1) return retval;
    retval += tset<value_type,1>(t, obj_iter); if (retval != 2) return retval;
    retval += tset<value_type,2>(t, obj_iter); if (retval != 3) return retval;
    retval += tset<value_type,3>(t, obj_iter); if (retval != 4) return retval;
    retval += tset<value_type,4>(t, obj_iter); if (retval != 5) return retval;
    retval += tset<value_type,5>(t, obj_iter); if (retval != 6) return retval;
    retval += tset<value_type,6>(t, obj_iter);
    return retval;
  }
  
  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
    l.append(boost::get<1>(t));
    l.append(boost::get<2>(t));
    l.append(boost::get<3>(t));
    l.append(boost::get<4>(t));
    l.append(boost::get<5>(t));
    l.append(boost::get<6>(t));
  }
};

template <typename T> struct TypeChooser<T,8> {
  typedef typename boost::tuple<T,T,T,T,T,T,T,T> value_type;
  
  static int set(value_type& t, PyObject* obj_iter) {
    int retval = 0;
    retval += tset<value_type,0>(t, obj_iter); if (retval != 1) return retval;
    retval += tset<value_type,1>(t, obj_iter); if (retval != 2) return retval;
    retval += tset<value_type,2>(t, obj_iter); if (retval != 3) return retval;
    retval += tset<value_type,3>(t, obj_iter); if (retval != 4) return retval;
    retval += tset<value_type,4>(t, obj_iter); if (retval != 5) return retval;
    retval += tset<value_type,5>(t, obj_iter); if (retval != 6) return retval;
    retval += tset<value_type,6>(t, obj_iter); if (retval != 7) return retval;
    retval += tset<value_type,7>(t, obj_iter);
    return retval;
  }
  
  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
    l.append(boost::get<1>(t));
    l.append(boost::get<2>(t));
    l.append(boost::get<3>(t));
    l.append(boost::get<4>(t));
    l.append(boost::get<5>(t));
    l.append(boost::get<6>(t));
    l.append(boost::get<7>(t));
  }
};

template <typename T> struct TypeChooser<T,9> {
  typedef typename boost::tuple<T,T,T,T,T,T,T,T,T> value_type;
  
  static int set(value_type& t, PyObject* obj_iter) {
    int retval = 0;
    retval += tset<value_type,0>(t, obj_iter); if (retval != 1) return retval;
    retval += tset<value_type,1>(t, obj_iter); if (retval != 2) return retval;
    retval += tset<value_type,2>(t, obj_iter); if (retval != 3) return retval;
    retval += tset<value_type,3>(t, obj_iter); if (retval != 4) return retval;
    retval += tset<value_type,4>(t, obj_iter); if (retval != 5) return retval;
    retval += tset<value_type,5>(t, obj_iter); if (retval != 6) return retval;
    retval += tset<value_type,6>(t, obj_iter); if (retval != 7) return retval;
    retval += tset<value_type,7>(t, obj_iter); if (retval != 8) return retval;
    retval += tset<value_type,8>(t, obj_iter);
    return retval;
  }
  
  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
    l.append(boost::get<1>(t));
    l.append(boost::get<2>(t));
    l.append(boost::get<3>(t));
    l.append(boost::get<4>(t));
    l.append(boost::get<5>(t));
    l.append(boost::get<6>(t));
    l.append(boost::get<7>(t));
    l.append(boost::get<8>(t));
  }
};

template <typename T> struct TypeChooser<T,10> {
  typedef typename boost::tuple<T,T,T,T,T,T,T,T,T,T> value_type;
  
  static int set(value_type& t, PyObject* obj_iter) {
    int retval = 0;
    retval += tset<value_type,0>(t, obj_iter); if (retval != 1) return retval;
    retval += tset<value_type,1>(t, obj_iter); if (retval != 2) return retval;
    retval += tset<value_type,2>(t, obj_iter); if (retval != 3) return retval;
    retval += tset<value_type,3>(t, obj_iter); if (retval != 4) return retval;
    retval += tset<value_type,4>(t, obj_iter); if (retval != 5) return retval;
    retval += tset<value_type,5>(t, obj_iter); if (retval != 6) return retval;
    retval += tset<value_type,6>(t, obj_iter); if (retval != 7) return retval;
    retval += tset<value_type,7>(t, obj_iter); if (retval != 8) return retval;
    retval += tset<value_type,8>(t, obj_iter); if (retval != 9) return retval;
    retval += tset<value_type,9>(t, obj_iter);
    return retval;
  }
  
  static void fill(const value_type& t, list& l) {
    l.append(boost::get<0>(t));
    l.append(boost::get<1>(t));
    l.append(boost::get<2>(t));
    l.append(boost::get<3>(t));
    l.append(boost::get<4>(t));
    l.append(boost::get<5>(t));
    l.append(boost::get<6>(t));
    l.append(boost::get<7>(t));
    l.append(boost::get<8>(t));
    l.append(boost::get<9>(t));
  }
};

/**
 * Objects of this type create a binding between boost::tuple<T,T,...> and
 * python iterables. You can specify a python iterable as a parameter to a
 * bound method that would normally receive a boost::tuple<T,T,...> or a const
 * boost::tuple<T,T,...>& and the conversion will just magically happen.
 */
template <typename T, int N>
struct tuple_from_sequence {

  typedef typename TypeChooser<T,N>::value_type container_type;
  
  /**
   * Registers converter from any python sequence into a boost::tuple<T,T,...>
   */
  tuple_from_sequence() {
    converter::registry::push_back(&convertible, &construct, 
        type_id<container_type>());
  }

  /**
   * This method will determine if the input python object is convertible into
   * a boost::tuple<T,T,...>
   *
   * Conditions:
   * - The input object has to have N elements (N = number of template params.)
   * - The input object has to be iterable.
   * - All elements in the input object have to be convertible to T objects
   */
  static void* convertible(PyObject* obj_ptr) {

    /**
     * this bit will check if the input obj is one of the expected input types
     * It will return 0 if the element in question is neither:
     * - a list
     * - a tuple
     * - an iterable
     * - a range
     * - is not a string _and_ is not an unicode string _and_ 
     *   (is a valid object pointer _or_ (too long to continue... ;-) 
     */
    if (!(PyList_Check(obj_ptr)
          || PyTuple_Check(obj_ptr)
          || PyIter_Check(obj_ptr)
          || PyRange_Check(obj_ptr)
          || ( !PyString_Check(obj_ptr)
            && !PyUnicode_Check(obj_ptr)
            && ( obj_ptr->ob_type == 0
              || obj_ptr->ob_type->ob_type == 0
              || obj_ptr->ob_type->ob_type->tp_name == 0
              || std::strcmp(
                obj_ptr->ob_type->ob_type->tp_name,
                "Boost.Python.class") != 0)
            && PyObject_HasAttrString(obj_ptr, "__len__")
            && PyObject_HasAttrString(obj_ptr, "__getitem__")))) return 0;

    //this bit will check if we have exactly N
    if(PyObject_Length(obj_ptr) != N) {
      PyErr_Clear();
      return 0;
    }

    //this bit will make sure we can extract an iterator from the object
    handle<> obj_iter(allow_null(PyObject_GetIter(obj_ptr)));
    if (!obj_iter.get()) { // must be convertible to an iterator
      PyErr_Clear();
      return 0;
    }

    //this bit will check every element for convertibility into "T"
    bool is_range = PyRange_Check(obj_ptr);
    std::size_t i=0;
    for(;;++i) { //if everything ok, should leave for loop with i == N
      handle<> py_elem_hdl(allow_null(PyIter_Next(obj_iter.get())));
      if (PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
      }
      if (!py_elem_hdl.get()) break; // end of iteration
      object py_elem_obj(py_elem_hdl);
      extract<T> elem_proxy(py_elem_obj);
      if (!elem_proxy.check()) return 0;
      if (is_range) break; // in a range all elements are of the same type
    }
    if (!is_range) assert(i == N);

    return obj_ptr;
  }

  /**
   * This method will finally construct the C++ element out of the python
   * object that was input. Please note that when boost::python reaches this
   * method, the object has already been checked for convertibility.
   */
  static void construct(PyObject* obj_ptr,
      converter::rvalue_from_python_stage1_data* data) {
    handle<> obj_iter(PyObject_GetIter(obj_ptr));
    void* storage = ((converter::rvalue_from_python_storage<container_type>*)data)->storage.bytes;
    new (storage) container_type();
    data->convertible = storage;
    container_type& result = *((container_type*)storage);
    int i = TypeChooser<T,N>::set(result, obj_iter.get());
    if (i != N) {
      boost::format s("expected %d elements for boost::tuple<T[N]>, got %d");
      s % N % N % i;
      PyErr_SetString(PyExc_RuntimeError, s.str().c_str());
      throw_error_already_set();
    }
  }

};

/**
 * Objects of this type bind boost::tuple<T,T,...> to python tuples. Your
 * method generates as output an object of this type and the object will be
 * automatically converted into a python tuple.
 */
template <typename T, int N>
struct tuple_to_tuple {
  
  typedef typename TypeChooser<T,N>::value_type container_type;

  static PyObject* convert(const container_type& tv) {
    list result;
    TypeChooser<T,N>::fill(tv, result);
    return incref(tuple(result).ptr());
  }

  static const PyTypeObject* get_pytype() { return &PyTuple_Type; }

};

template <typename T, int N>
void register_tuple_to_tuple() {

  typedef typename TypeChooser<T,N>::value_type container_type;

  to_python_converter<container_type, 
    tuple_to_tuple<T, N>
#if defined BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
      ,true
#endif
      >();
}

void bind_core_tuple () {

  /**
   * The following struct constructors will make sure we can input
   * boost::tuple<T,T,...> in our bound C++ routines w/o needing to specify
   * special converters each time. The rvalue converters allow boost::python to
   * automatically map the following inputs:
   *
   * a) const boost::tuple<T,T,...>& (pass by const reference)
   * b) boost::tuple<T,T,...> (pass by value)
   *
   * Please note that the last case:
   * 
   * c) boost::tuple<T,T,...>& (pass by non-const reference)
   *
   * is NOT covered by these converters. The reason being that because the
   * object may be changed, there is no way for boost::python to update the
   * original python object, in a sensible manner, at the return of the method.
   *
   * Avoid passing by non-const reference in your methods.
   */
# define BOOST_PP_LOCAL_LIMITS (1, 10)
# define BOOST_PP_LOCAL_MACRO(D) \
  tuple_from_sequence<bool,D>(); \
  tuple_from_sequence<int8_t,D>(); \
  tuple_from_sequence<int16_t,D>(); \
  tuple_from_sequence<int32_t,D>(); \
  tuple_from_sequence<int64_t,D>(); \
  tuple_from_sequence<uint8_t,D>(); \
  tuple_from_sequence<uint16_t,D>(); \
  tuple_from_sequence<uint32_t,D>(); \
  tuple_from_sequence<uint64_t,D>(); \
  tuple_from_sequence<float,D>(); \
  tuple_from_sequence<double,D>(); \
  tuple_from_sequence<std::complex<float>,D>(); \
  tuple_from_sequence<std::complex<double>,D>();
# include BOOST_PP_LOCAL_ITERATE()

  /**
   * The following struct constructors will make C++ return values of type
   * blitz::TinyVector<T,N> to show up in the python side as tuples.
   */
# define BOOST_PP_LOCAL_LIMITS (1, 10)
# define BOOST_PP_LOCAL_MACRO(D) \
  register_tuple_to_tuple<bool,D>(); \
  register_tuple_to_tuple<int8_t,D>(); \
  register_tuple_to_tuple<int16_t,D>(); \
  register_tuple_to_tuple<int32_t,D>(); \
  register_tuple_to_tuple<int64_t,D>(); \
  register_tuple_to_tuple<uint8_t,D>(); \
  register_tuple_to_tuple<uint16_t,D>(); \
  register_tuple_to_tuple<uint32_t,D>(); \
  register_tuple_to_tuple<uint64_t,D>(); \
  register_tuple_to_tuple<float,D>(); \
  register_tuple_to_tuple<double,D>(); \
  register_tuple_to_tuple<std::complex<float>,D>(); \
  register_tuple_to_tuple<std::complex<double>,D>();
# include BOOST_PP_LOCAL_ITERATE()

}
