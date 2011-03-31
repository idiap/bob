/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 11 Mar 12:02:35 2011 
 *
 * @brief Shared tools for array indexing
 */

#ifndef TORCH_CORE_PYTHON_ARRAY_INDEXING_H 
#define TORCH_CORE_PYTHON_ARRAY_INDEXING_H

#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <boost/format.hpp>
#include <blitz/array.h>
#include <blitz/tinyvec-et.h>

namespace Torch { namespace python {

  namespace range {
    extern int fromStart;
    extern int toEnd;
  }

  /**
   * Returns true or false if the index is inside the given bounds or not.
   */
  bool check_index(int index, int base, int extent);

  /**
   * Checks a single range for the indexes it tries to address. Raises a proper
   * python exception if that is not ok.
   */
  void check_range(int dimension, const blitz::Range& r, int base, int extent);

  /**
   * Checks a single index. Raises a proper exception if that is not ok.
   */
  int check_range(int dimension, int index, int base, int extent);

  /**
   * Converts a slice object into a range
   */
  void slice2range(const boost::python::slice& s, blitz::Range& r);

  /**
   * Checks two sizes for equality, if they are different, throw a nice
   * exception showing the case
   */
  template <typename T, int N> void check_size(const blitz::Array<T,N>& a, 
      int dim, const blitz::Range& r, int other_size) {
    int rec_size = r.last(a.ubound(dim)) - r.first(a.lbound(dim)) + 1;
    if (rec_size != other_size) {
      boost::format msg("Subarray setting with array of different sizes on dimension %d. Assigned subarray has size %d and input data has size %d (for the given dimension)");
      msg % dim % rec_size % other_size;
      PyErr_SetString(PyExc_IndexError, msg.str().c_str());
      boost::python::throw_error_already_set();
    }
  }

  /**
   * fills a1 with data from a2, if sizes match
   */
  template <typename T, int N> void fill
    (blitz::Array<T,N>& a1, const blitz::Array<T,N>& a2) {
      if (blitz::any(a1.shape() != a2.shape())) {
        boost::format msg("Assignment using arrays with different sizes %s != %s");
        boost::python::str bs1(boost::python::tuple(a1.extent()));
        std::string s1 = boost::python::extract<std::string>(bs1);
        boost::python::str bs2(boost::python::tuple(a2.extent()));
        std::string s2 = boost::python::extract<std::string>(bs2);
        msg % s1 % s2;
        PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
        boost::python::throw_error_already_set();
      }
      a1 = a2; //simple copy through blitz standard c++ operator
    }

}}

#endif /* ARRAY_INDEXING_H */

