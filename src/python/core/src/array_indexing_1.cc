/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 11 Mar 12:09:36 2011 
 *
 * @brief Implements specificities of 1-D array indexing
 */

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"
#include "core/python/array_indexing.h"

namespace tp = Torch::python;
namespace bp = boost::python;
namespace tca = Torch::core::array;

/**
 * The methods here implement the __getitem__ functionality expected by every
 * python random access container. Here is their behavior:
 * 1. Checks if the received index is a valid entry w.r.t. N and the size of
 * the input array. Here are the possibilities
 *    a) If index is a single integer, N has to be 1, the index has to be
 *    smaller than the first array extent. The same is applicable if index is
 *    a single slice element.
 *    b) If index is a tuple composed of integers, N == len(index) and every
 *    index has to be smaller than the extent they refer to. The same is
 *    applicable if the tuple is composed of a mix of integers and slices.
 * 2. If the input index refers to a single element of the array, we return
 * this value as a python object. If the input index refers to a slice, we
 * return a new array referencing the array in the positions selected.
 *
 * Negative indexing is supported, mimmicking normal python random access
 * containers.
 */

template <typename T> static bp::object getset 
(blitz::Array<T,1>& a, bp::object index, bp::object value=bp::object()) {

  bp::object consider = index;

  //the input indexer may be a tuple, in which case must have length 1
  bp::extract<bp::tuple> tuple_check(index);
  if (tuple_check.check()) {
    bp::tuple tindex = tuple_check();
    if (bp::len(tindex) != 1) {
      boost::format msg("Trying to index blitz::Array<T,1> using a tuple with %d entries");
      msg % bp::len(tindex);
      PyErr_SetString(PyExc_IndexError, msg.str().c_str());
      boost::python::throw_error_already_set();
    }
    consider = tindex[0];
  }

  //consider now holds the index to the 1-D blitz array. Let's see what it is
  bp::extract<blitz::Range> r1c(consider);
  bp::extract<int> i1c(consider);

  //switch through the different possibilities we must cover
  if (r1c.check()) {
    //ERROR: Cannot do this twice in a row: python crashes!
    blitz::Range r1 = r1c(); 
    tp::check_range(0, r1, a.lbound(0), a.extent(0));
    bp::object retval(a(r1));
    if (value.ptr() != Py_None) retval.attr("fill")(value);
    return retval;
  }
  if (i1c.check()) {
    int r1 = i1c();
    r1 = tp::check_range(0, r1, a.lbound(0), a.extent(0));
    if (value.ptr() != Py_None) a(r1) = bp::extract<T>(value);
    return bp::object(a(r1));
  }

  //if you get to this point, have to raise
  boost::format msg("Cannot index blitz::Array<T,1> with input arguments");
  PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
  boost::python::throw_error_already_set();
  return bp::object(); //shuts up gcc
}

template <typename T>
static bp::object getitem (blitz::Array<T,1>& a, bp::object index) {
  return getset(a, index);
}

template <typename T> static bp::object setitem 
(blitz::Array<T,1>& a, bp::object index, bp::object value) {
  return getset(a, index, value);
}

template <typename T>
static void bind_indexing_1(tp::array<T,1>& array) {
  array.object()->def("fill", &tp::fill<T,1>, (bp::arg("self"), bp::arg("other")), "Sets the contents of the current array to be the same as the other array. A full size check is a precondition and I'll raise an exception if the destination sizes and source sizes do not properly match.");
  array.object()->def("__getitem__", &getitem<T>, (bp::arg("self"), bp::arg("index")), "Accesses one element of the array.");
  array.object()->def("__setitem__", &setitem<T>, (bp::arg("self"), bp::arg("index"), bp::arg("value")), "Sets one element of the array.");
}

void bind_array_indexing_1 () {
  bind_indexing_1(tp::bool_1);
  bind_indexing_1(tp::int8_1);
  bind_indexing_1(tp::int16_1);
  bind_indexing_1(tp::int32_1);
  bind_indexing_1(tp::int64_1);
  bind_indexing_1(tp::uint8_1);
  bind_indexing_1(tp::uint16_1);
  bind_indexing_1(tp::uint32_1);
  bind_indexing_1(tp::uint64_1);
  bind_indexing_1(tp::float32_1);
  bind_indexing_1(tp::float64_1);
  //bind_indexing_1(tp::float128_1);
  bind_indexing_1(tp::complex64_1);
  bind_indexing_1(tp::complex128_1);
  //bind_indexing_1(tp::complex256_1);
}
