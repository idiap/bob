/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 27 Sep 03:18:30 2011
 *
 * @brief A set of example blitz::Array <-> NumPy wrappers
 */

#include <blitz/array.h>
#include <stdint.h>

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = Torch::python;

/**
 * These are example methods we will bind to python. All other use-cases will
 * fall under one of these situations or a combination of these situations. Use
 * what you have learned in each to find the perfect way to bind your methods.
 */

/**
 * Example overloading with templates. In this example the input array is only
 * used for "consultation" and its data is not kept anywhere.
 */
template <typename T, int N>
int count_elements_bzref(const blitz::Array<T,N>& in) {
  return in.size();
}

/**
 * Example returning a new array
 */
template <typename T>
blitz::Array<T,2> zeroes_2d(int rows, int cols) {
  blitz::Array<T,2> retval(rows, cols);
  retval = 0;
  return retval;
}

/**
 * Example array transformation with memory allocation. The input array is only
 * looked at and the transformed data is set on another array.
 */
template <typename T, int N>
blitz::Array<T,N> add_1_alloc(const blitz::Array<T,N>& in) {
  blitz::Array<T,N> retval = in.copy();
  retval += 1;
  return retval;
}

/**
 * Example array transformation without memory allocation. The output array is
 * not dynamically allocated, but provided by the user.
 */
template <typename T, int N>
void add_1(const blitz::Array<T,N>& in, blitz::Array<T,N>& out) {
  //you may want to check if "out" conforms, but for our example, we skip it.
  out = in + 1;
}

/**
 * Pointer to constant reference
 */
static blitz::Array<double,2> TEST_DATA(10,10);

/**
 * Accesses the static variable. This method examplifies typical use when
 * returning in-class private attributes to the user.
 */
const blitz::Array<double,2>& get_const_data() {
  return TEST_DATA;
}

/**
 * These are ways to bind the above methods to python. By default, our stock
 * bindings will provide the following:
 *
 * a) Parameter converter to const blitz::Array<T,N>& or blitz::Array<T,N>.
 *    Note: Avoid at all costs parameter passing by value! You will not be able
 *    to generate a reliable python binding in this case, so just don't do it
 *    in C++ as well. 
 *
 *    For the automated converter, if you bind your method w/o any special
 *    regards, parameters originally set to receive a const blitz::Array<T,N>&
 *    will be efficiently (i.e., as efficently as possible) converted from any
 *    possible python object to the destination array.
 *
 *    Conversion rules:
 *
 *    a.1) Python iterables will be converted to a const blitz::Array<T,N>& by
 *    copy. This is the slowest of the possible situations as the data has to
 *    be converted into an array and then cast to the desired type, but it is
 *    very convinient to the end user.
 *
 *    a.2) numpy.ndarrays will be cast/copied if they don't have a perfect
 *    data type match to the required type T of the blitz::Array.
 *
 *    a.3) numpy.ndarrays will be referenced (no data copy) if they have a
 *    perfect data type match to the required type T of the blitz::Array. This
 *    is by far the fastest.
 *
 * b) Return value converter for "blitz::Array<T,N>". In this case the data is
 *    always copied to a new numpy.ndarray object. If you want to avoid the
 *    copy, you have to pass the output array by non-const reference at your
 *    method and fiddle wit it yourself as shown below.
 *
 * Binding non-const references is a little bit more tricky as you want the
 * user data to be updated. This is the trick: we overlay a blitz::Array layer
 * on the top of a numpy.ndarray and pass that to our C++ method.
 */

object py_zeroes_2d(int rows, int cols, object dtype=object()) {
  tp::dtype dt(dtype);
  switch(dt.type()) {
    case NPY_FLOAT32:
      return object(zeroes_2d<float>(rows, cols));
    case NPY_FLOAT64:
      return object(zeroes_2d<double>(rows,cols));
    default:
      PYTHON_ERROR(TypeError, "no support for data type");
  }
}
BOOST_PYTHON_FUNCTION_OVERLOADS(py_zeroes_2d_overloads, py_zeroes_2d, 2, 3)

template <typename T, int N>
void py_add_1(const blitz::Array<T,N>& in, numeric::array& out) {
  //a call to tp::numpy_bz<T,N>() will create a thin blitz::Array wrapper
  //around the original numpy.ndarray object.
  blitz::Array<T,N> tmp = tp::numpy_bz<T,N>(out);
  add_1(in, tmp);
}

void bind_core_array_examples() {

  /**
   * To bind simple functions as one receiving const blitz::Array<>& and
   * returning simple objects, you just have to declare them w/o any special
   * considerations.
   */
  def("count_elements_bzref", &count_elements_bzref<float,2>, (arg("input")), "Returns the total element count for an array");
  def("count_elements_bzref", &count_elements_bzref<double,2>, (arg("input")), "Returns the total element count for an array");

  /**
   * To bind a method that returns a new array, we copy the data. This example
   * is not really a nice idea, because you can do that more efficiently in
   * numpy anyway, but it shows you how to wrap a method that returns by value.
   *
   * There are no special requirements for this, to-python automatic converters
   * will just handle it correctly. You should just declare as many template
   * variants as you need in python.
   *
   * Note that, in this example, overloading will *not* work as you expect
   * since the automatic overloading mechanism cannot decide, from the input
   * arguments, which of the variants to call. You have two options:
   *
   * 1. You create each function with a special name extension attached to
   * them like bellow.
   *
   * In python, you write a function "zeroes_2d" that receives a dtype
   * parameter and chooses dynamically which of the two to call bellow.
   */
  def("zeroes_2d_float32", &zeroes_2d<float>, (arg("rows"), arg("cols")), "Creates a float32-matrix filled with zeroes");
  def("zeroes_2d_float64", &zeroes_2d<double>, (arg("rows"), arg("cols")), "Creates a float64-matrix filled with zeroes");

  /**
   * The second option is to write a method that takes a "dtype" argument as
   * input parameter and decides what is the correct C++ function to call.
   *
   * Look at the implementation of py_zeroes_2d() above for more details. The
   * last parameter is optional.
   */
  def("zeroes_2d", (object(*)(int,int,object))0, py_zeroes_2d_overloads((arg("rows"), arg("cols"), arg("dtype")=object()), "Creates a 2d matrix filled with zeroes -- user can choose the data type by passing any valid numpy.dtype representation. By default, if dtype is not specified, used the numpy default (normally float64/double)."));

  /**
   * To bind a method that returns a (newly allocated) blitz::Array<>, you
   * should do it simply as well. Template overloading can be achieved in two
   * ways in this mode. The first, is to declare all combinations exhaustively
   * like we do it bellow.
   *
   * Please note that, when you do this, you are asking for auto-conversion of
   * numpy.ndarray's into blitz::Array<> and this goes as the conversion rules
   * explained above.
   *
   * We are using boost::python automatic overloading resolution in this way,
   * which means you will have to pass a precise type or the wrong function may
   * be called, depending on the order you declare the extensions -- the last
   * declaration is evaluated first. Think of it as a first-in-last-out stack.
   *
   * Note we have setup the automatic conversion like this: if you pass a
   * numpy.ndarray, that is take as a precise type. If you pass an iterable to
   * be converted, the call that first satisfies the precise type
   * convertibility will be done.
   *
   * @warning: This way of wrapping overloaded template functions may result in
   * code bloating. It works well if you only want to bind a few types, but may
   * result in code bloat if you need to bind a lot. It may be unavoidable if
   * you plan to support many types though ;-)
   */
  def("add_1_alloc", &add_1_alloc<double,1>, (arg("input")), "Adds 1 to all elements in the array and returns a new copy");
  def("add_1_alloc", &add_1_alloc<uint32_t,1>, (arg("input")), "Adds 1 to all elements in the array and returns a new copy");

  /**
   * To bind a method that receives a non-const blitz::Array<T,N> (one that
   * will be modified within the class), we need to wrap it using a provided
   * ndarray extension. This extension is auto-converted to a blitz::Array
   * within the method. We write the overloading like for the const reference
   * case.
   */
  def("add_1", &py_add_1<double,1>, (arg("input"), arg("output")), "Adds 1 to all elements in the array and returns a new copy");
  def("add_1", &py_add_1<uint32_t,1>, (arg("input"), arg("output")), "Adds 1 to all elements in the array and returns a new copy");

  /**
   * Finally, we tackle how to return const blitz::Array<>&'s in a way that
   * avoids copying. 
   */

}
