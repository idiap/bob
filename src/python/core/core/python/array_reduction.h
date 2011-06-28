/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Fri Apr  1 15:52:47 CEST 2011
 *
 * @brief Describes a few possible array reductions
 */

#ifndef TORCH_PYTHON_CORE_ARRAY_REDUCTION_H 
#define TORCH_PYTHON_CORE_ARRAY_REDUCTION_H

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"
#include "core/python/exception.h"

namespace Torch { namespace python {

  /**
   * According to the Blitz++ manual, partial reductions can only be done over
   * the last dimension. So, we propose equivalent methods that reduce over
   * such last dimension (named *_r). Partial reductions will work for any
   * number of array dimensionality N >= 2.
   */

  template <typename T, int N> T sum(blitz::Array<T,N>& i) { return blitz::sum(i); }

  template <typename T, int N>
  struct sum_r {
    static blitz::Array<T,N-1> op (const blitz::Array<T,N>& i) {
      return blitz::Array<T,N-1>(blitz::sum(i, blitz::IndexPlaceholder<N-1>()));
    }
  };

  template <typename T>
  struct sum_r<T,1> {
    static T op (const blitz::Array<T,1>& i) {
      return blitz::sum(i);
    }
  };

  template <typename T, int N> T product(blitz::Array<T,N>& i) { return blitz::product(i); }

  template <typename T, int N>
  struct product_r {
    static blitz::Array<T,N-1> op (const blitz::Array<T,N>& i) {
      return blitz::Array<T,N-1>(blitz::product(i, blitz::IndexPlaceholder<N-1>()));
    }
  };

  template <typename T>
  struct product_r<T,1> {
    static T op (const blitz::Array<T,1>& i) {
      return blitz::product(i);
    }
  };

  template <typename T, int N> T mean(blitz::Array<T,N>& i) { return blitz::mean(i); }

  template <typename T, int N>
  struct mean_r {
    static blitz::Array<T,N-1> op (const blitz::Array<T,N>& i) {
      return blitz::Array<T,N-1>(blitz::mean(i, blitz::IndexPlaceholder<N-1>()));
    }
  };

  template <typename T>
  struct mean_r<T,1> {
    static T op (const blitz::Array<T,1>& i) {
      return blitz::mean(i);
    }
  };

  template <typename T, int N> T min(blitz::Array<T,N>& i) { return blitz::min(i); }

  template <typename T, int N>
  struct min_r {
    static blitz::Array<T,N-1> op (const blitz::Array<T,N>& i) {
      return blitz::Array<T,N-1>(blitz::min(i, blitz::IndexPlaceholder<N-1>()));
    }
  };

  template <typename T>
  struct min_r<T,1> {
    static T op (const blitz::Array<T,1>& i) {
      return blitz::min(i);
    }
  };

  template <typename T, int N> T max(blitz::Array<T,N>& i) { return blitz::max(i); }

  template <typename T, int N>
  struct max_r {
    static blitz::Array<T,N-1> op (const blitz::Array<T,N>& i) {
      return blitz::Array<T,N-1>(blitz::max(i, blitz::IndexPlaceholder<N-1>()));
    }
  };

  template <typename T>
  struct max_r<T,1> {
    static T op (const blitz::Array<T,1>& i) {
      return blitz::max(i);
    }
  };

  template <typename T, int N> blitz::TinyVector<int,N> minIndex(blitz::Array<T,N>& i) { return blitz::minIndex(i); }

  template <typename T, int N> blitz::TinyVector<int,N> maxIndex(blitz::Array<T,N>& i) { return blitz::maxIndex(i); }

  template <typename T, int N> int count(blitz::Array<T,N>& i) { return blitz::count(i); }

  template <typename T, int N>
  struct count_r {
    static blitz::Array<int,N-1> op (const blitz::Array<T,N>& i) {
      return blitz::Array<int,N-1>(blitz::count(i, blitz::IndexPlaceholder<N-1>()));
    }
  };

  template <typename T>
  struct count_r<T,1> {
    static int op (const blitz::Array<T,1>& i) {
      return blitz::count(i);
    }
  };

  template <typename T, int N> bool any(blitz::Array<T,N>& i) { return blitz::any(i); }

  template <typename T, int N>
  struct any_r {
    static blitz::Array<bool,N-1> op (const blitz::Array<T,N>& i) {
      return blitz::Array<bool,N-1>(blitz::any(i, blitz::IndexPlaceholder<N-1>()));
    }
  };

  template <typename T>
  struct any_r<T,1> {
    static bool op (const blitz::Array<T,1>& i) {
      return blitz::any(i);
    }
  };

  template <typename T, int N> bool all(blitz::Array<T,N>& i) { return blitz::all(i); }
  
  template <typename T, int N>
  struct all_r {
    static blitz::Array<bool,N-1> op (const blitz::Array<T,N>& i) {
      return blitz::Array<bool,N-1>(blitz::all(i, blitz::IndexPlaceholder<N-1>()));
    }
  };

  template <typename T>
  struct all_r<T,1> {
    static bool op (const blitz::Array<T,1>& i) {
      return blitz::all(i);
    }
  };

  /**
   * Common methods
   */
  template <typename T, int N> void bind_common_reductions 
    (Torch::python::array<T,N>& array) {
      array.object()->def("sum", &sum<T,N>, "Summation");
      array.object()->def("partial_sum", &sum_r<T,N>::op, "Partial Summation over the last dimension");
      array.object()->def("product", &product<T,N>, "Product");
      array.object()->def("partial_product", &product_r<T,N>::op, "Partial Product over the last dimension");
  }

  /**
   * Non-complex methods
   */
  template <typename T, int N> void bind_reductions (Torch::python::array<T,N>& array) {
    bind_common_reductions(array); 
    array.object()->def("mean", &mean<T,N>, "Arithmetic mean");
    array.object()->def("partial_mean", &mean_r<T,N>::op, "Partial arithmetic mean over the last dimension");
    array.object()->def("min", &min<T,N>, "Minimum value");
    array.object()->def("partial_min", &min_r<T,N>::op, "Minimum value over the last dimension");
    array.object()->def("max", &max<T,N>, "Maximum value");
    array.object()->def("partial_max", &max_r<T,N>::op, "Maximum value over the last dimension");
    array.object()->def("minIndex", &minIndex<T,N>, "Index of the minimum value (returns tuple.");
    array.object()->def("maxIndex", &maxIndex<T,N>, "Index of the maximum value (returns tuple.");
    array.object()->def("any", &any<T,N>, "True if the array is True anywhere.");
    array.object()->def("partial_any", &any_r<T,N>::op, "True if the array is True anywhere over the last dimension");
    array.object()->def("all", &all<T,N>, "True if the array is True everywhere.");
    array.object()->def("partial_all", &all_r<T,N>::op, "True if the array is True everywhere over the last dimension");
    array.object()->def("count", &count<T,N>, "Counts the number of times the expression is true anywhere.");
    array.object()->def("partial_count", &count_r<T,N>::op, "Counts the number of evaluated 'true's over the last dimension.");
  }

}}

#endif /* TORCH_PYTHON_CORE_ARRAY_REDUCTION_H */
