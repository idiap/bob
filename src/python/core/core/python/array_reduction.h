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

namespace Torch { namespace python {

  //some reductions
  //TODO: Missing reductions that take a dimension parameter (please note this
  //is not an "int". Blitz provides its own scheme with indexes which are fully
  //fledged types. See the manual.

  template <typename T, int N> T sum(blitz::Array<T,N>& i) { return blitz::sum(i); }
  //template <typename T, int N> blitz::Array<T,1> sum_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::sum(i, dim)); }

  template <typename T, int N> T product(blitz::Array<T,N>& i) { return blitz::product(i); }
  //template <typename T, int N> blitz::Array<T,1> product_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::product(i, dim)); }

  template <typename T, int N> T mean(blitz::Array<T,N>& i) { return blitz::mean(i); }
  //template <typename T, int N> blitz::Array<T,1> mean_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::mean(i, dim)); }

  template <typename T, int N> T min(blitz::Array<T,N>& i) { return blitz::min(i); }
  //template <typename T, int N> blitz::Array<T,1> min_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::min(i, dim)); }

  template <typename T, int N> T max(blitz::Array<T,N>& i) { return blitz::max(i); }
  //template <typename T, int N> blitz::Array<T,1> max_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::max(i, dim)); }

  template <typename T, int N> blitz::TinyVector<int,N> minIndex(blitz::Array<T,N>& i) { return blitz::minIndex(i); }
  //template <typename T, int N> blitz::TinyVector<int,N> minIndex_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::minIndex(i, dim)); }

  template <typename T, int N> blitz::TinyVector<int,N> maxIndex(blitz::Array<T,N>& i) { return blitz::maxIndex(i); }
  //template <typename T, int N> blitz::TinyVector<int,N> maxIndex_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::maxIndex(i, dim)); }

  template <typename T, int N> int count(blitz::Array<T,N>& i) { return blitz::count(i); }
  //template <typename T, int N> blitz::Array<int,1> count_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::count(i, dim)); }

  template <typename T, int N> bool any(blitz::Array<T,N>& i) { return blitz::any(i); }
  //template <typename T, int N> blitz::Array<bool,1> any_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::any(i, dim)); }

  template <typename T, int N> bool all(blitz::Array<T,N>& i) { return blitz::all(i); }
  //template <typename T, int N> blitz::Array<bool,1> all_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::all(i, dim)); }

  /**
   * Common methods
   */
  template <typename T, int N> void bind_common_reductions 
    (Torch::python::array<T,N>& array) {
      array.object()->def("sum", &sum<T,N>, "Summation");
      array.object()->def("product", &product<T,N>, "Product");
  }

  /**
   * Non-complex methods
   */
  template <typename T, int N> void bind_reductions (Torch::python::array<T,N>& array) {
    bind_common_reductions(array); 
    array.object()->def("mean", &mean<T,N>, "Arithmetic mean");
    array.object()->def("min", &min<T,N>, "Minimum value");
    array.object()->def("max", &max<T,N>, "Maximum value");
    array.object()->def("minIndex", &minIndex<T,N>, "Index of the minimum value (returns tuple.");
    array.object()->def("maxIndex", &maxIndex<T,N>, "Index of the maximum value (returns tuple.");
    array.object()->def("any", &any<T,N>, "True if the array is True anywhere.");
    array.object()->def("all", &all<T,N>, "True if the array is True everywhere.");
    array.object()->def("count", &count<T,N>, "Counts the number of times the expression is true anywhere.");
  }

}}

#endif /* TORCH_PYTHON_CORE_ARRAY_REDUCTION_H */
