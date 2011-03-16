/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed  9 Mar 19:51:00 2011 
 *
 * @brief Implements specificities of 4-D array indexing
 */

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"
#include "core/python/array_indexing.h"

namespace tp = Torch::python;
namespace bp = boost::python;

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
(blitz::Array<T,4>& a, bp::tuple index, bp::object value=bp::object()) {
  if (bp::len(index) != 4) {
    boost::format msg("Trying to index blitz::Array<T,4> using a tuple with %d entries");
    msg % bp::len(index);
    PyErr_SetString(PyExc_IndexError, msg.str().c_str());
    boost::python::throw_error_already_set();
  }
  
  bp::extract<blitz::Range> r1c(index[0]);
  bp::extract<blitz::Range> r2c(index[1]);
  bp::extract<blitz::Range> r3c(index[2]);
  bp::extract<blitz::Range> r4c(index[3]);
  bp::extract<int> i1c(index[0]);
  bp::extract<int> i2c(index[1]);
  bp::extract<int> i3c(index[2]);
  bp::extract<int> i4c(index[3]);

  //switch through the different possibilities we must cover
  if (r1c.check()) {
    blitz::Range r1 = r1c(); 
    tp::check_range(0, r1, a.lbound(0), a.extent(0));
    if (r2c.check()) {
      blitz::Range r2 = r2c(); 
      tp::check_range(1, r2, a.lbound(1), a.extent(1));
      if (r3c.check()) {
        blitz::Range r3 = r3c(); 
        tp::check_range(2, r3, a.lbound(2), a.extent(2));
        if (r4c.check()) {
          blitz::Range r4 = r4c(); 
          tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
        if (i4c.check()) {
          int r4 = i4c();
          r4 = tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
      }
      if (i3c.check()) {
        int r3 = i3c();
        r3 = tp::check_range(2, r3, a.lbound(2), a.extent(2));
        if (r4c.check()) {
          blitz::Range r4 = r4c(); 
          tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
        if (i4c.check()) {
          int r4 = i4c();
          r4 = tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
      }
    }
    if (i2c.check()) {
      int r2 = i2c();
      r2 = tp::check_range(1, r2, a.lbound(1), a.extent(1));
      if (r3c.check()) {
        blitz::Range r3 = r3c(); 
        tp::check_range(2, r3, a.lbound(2), a.extent(2));
        if (r4c.check()) {
          blitz::Range r4 = r4c(); 
          tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
        if (i4c.check()) {
          int r4 = i4c();
          r4 = tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
      }
      if (i3c.check()) {
        int r3 = i3c();
        r3 = tp::check_range(2, r3, a.lbound(2), a.extent(2));
        if (r4c.check()) {
          blitz::Range r4 = r4c(); 
          tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
        if (i4c.check()) {
          int r4 = i4c();
          r4 = tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
      }
    }
  }
  if (i1c.check()) {
    int r1 = i1c();
    r1 = tp::check_range(0, r1, a.lbound(0), a.extent(0));
    if (r2c.check()) {
      blitz::Range r2 = r2c(); 
      tp::check_range(1, r2, a.lbound(1), a.extent(1));
      if (r3c.check()) {
        blitz::Range r3 = r3c(); 
        tp::check_range(2, r3, a.lbound(2), a.extent(2));
        if (r4c.check()) {
          blitz::Range r4 = r4c(); 
          tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
        if (i4c.check()) {
          int r4 = i4c();
          r4 = tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
      }
      if (i3c.check()) {
        int r3 = i3c();
        r3 = tp::check_range(2, r3, a.lbound(2), a.extent(2));
        if (r4c.check()) {
          blitz::Range r4 = r4c(); 
          tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
        if (i4c.check()) {
          int r4 = i4c();
          r4 = tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
      }
    }
    if (i2c.check()) {
      int r2 = i2c();
      r2 = tp::check_range(1, r2, a.lbound(1), a.extent(1));
      if (r3c.check()) {
        blitz::Range r3 = r3c(); 
        tp::check_range(2, r3, a.lbound(2), a.extent(2));
        if (r4c.check()) {
          blitz::Range r4 = r4c(); 
          tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
        if (i4c.check()) {
          int r4 = i4c();
          r4 = tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
      }
      if (i3c.check()) {
        int r3 = i3c();
        r3 = tp::check_range(2, r3, a.lbound(2), a.extent(2));
        if (r4c.check()) {
          blitz::Range r4 = r4c(); 
          tp::check_range(3, r4, a.lbound(3), a.extent(3));
          bp::object retval(a(r1,r2,r3,r4));
          if (!value.is_none()) retval.attr("fill")(value);
          return retval;
        }
        if (i4c.check()) {
          int r4 = i4c();
          r4 = tp::check_range(3, r4, a.lbound(3), a.extent(3));
          if (!value.is_none()) a(r1,r2,r3,r4) = bp::extract<T>(value);
          return bp::object(a(r1,r2,r3,r4));
        }
      }
    }
  }

  //if you get to this point, have to raise
  boost::format msg("Cannot index blitz::Array<T,4> with input arguments");
  PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
  boost::python::throw_error_already_set();
  return bp::object(); //shuts up gcc
}

template <typename T>
static bp::object getitem (blitz::Array<T,4>& a, bp::tuple index) {
  return getset(a, index);
}

template <typename T> static bp::object setitem 
(blitz::Array<T,4>& a, bp::tuple index, bp::object value) {
  return getset(a, index, value);
}

template <typename T>
static void bind_indexing_4(tp::array<T,4>& array) {
  array.object()->def("fill", &tp::fill<T,4>, (bp::arg("self"), bp::arg("other")), "Sets the contents of the current array to be the same as the other array. A full size check is a precondition and I'll raise an exception if the destination sizes and source sizes do not properly match.");
  array.object()->def("__getitem__", &getitem<T>, (bp::arg("self"), bp::arg("index")), "Accesses one element of the array.");
  array.object()->def("__setitem__", &setitem<T>, (bp::arg("self"), bp::arg("index"), bp::arg("value")), "Sets one or more elements of the array."); 
}

void bind_array_indexing_4 () {
  bind_indexing_4(tp::bool_4);
  bind_indexing_4(tp::int8_4);
  bind_indexing_4(tp::int16_4);
  bind_indexing_4(tp::int32_4);
  bind_indexing_4(tp::int64_4);
  bind_indexing_4(tp::uint8_4);
  bind_indexing_4(tp::uint16_4);
  bind_indexing_4(tp::uint32_4);
  bind_indexing_4(tp::uint64_4);
  bind_indexing_4(tp::float32_4);
  bind_indexing_4(tp::float64_4);
  //bind_indexing_4(tp::float128_4);
  bind_indexing_4(tp::complex64_4);
  bind_indexing_4(tp::complex128_4);
  //bind_indexing_4(tp::complex256_4);
}
