/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 11 Mar 23:24:51 2011 
 *
 * @brief Reductions 
 */

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"

namespace tp = Torch::python;
namespace bp = boost::python;

//some reductions
//TODO: Missing reductions that take a dimension parameter (please note this
//is not an "int". Blitz provides its own scheme with indexes which are fully
//fledged types. See the manual.

template <typename T, int N> static T sum(blitz::Array<T,N>& i) { return blitz::sum(i); }
//template <typename T, int N> static blitz::Array<T,1> sum_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::sum(i, dim)); }

template <typename T, int N> static T product(blitz::Array<T,N>& i) { return blitz::product(i); }
//template <typename T, int N> static blitz::Array<T,1> product_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::product(i, dim)); }

template <typename T, int N> static T mean(blitz::Array<T,N>& i) { return blitz::mean(i); }
//template <typename T, int N> static blitz::Array<T,1> mean_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::mean(i, dim)); }

template <typename T, int N> static T min(blitz::Array<T,N>& i) { return blitz::min(i); }
//template <typename T, int N> static blitz::Array<T,1> min_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::min(i, dim)); }

template <typename T, int N> static T max(blitz::Array<T,N>& i) { return blitz::max(i); }
//template <typename T, int N> static blitz::Array<T,1> max_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::max(i, dim)); }

template <typename T, int N> static blitz::TinyVector<int,N> minIndex(blitz::Array<T,N>& i) { return blitz::minIndex(i); }
//template <typename T, int N> static blitz::TinyVector<int,N> minIndex_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::minIndex(i, dim)); }

template <typename T, int N> static blitz::TinyVector<int,N> maxIndex(blitz::Array<T,N>& i) { return blitz::maxIndex(i); }
//template <typename T, int N> static blitz::TinyVector<int,N> maxIndex_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::maxIndex(i, dim)); }

template <typename T, int N> static int count(blitz::Array<T,N>& i) { return blitz::count(i); }
//template <typename T, int N> static blitz::Array<int,1> count_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::count(i, dim)); }

template <typename T, int N> static bool any(blitz::Array<T,N>& i) { return blitz::any(i); }
//template <typename T, int N> static blitz::Array<bool,1> any_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::any(i, dim)); }

template <typename T, int N> static bool all(blitz::Array<T,N>& i) { return blitz::all(i); }
//template <typename T, int N> static blitz::Array<bool,1> all_dim(blitz::Array<T,N>& i, int dim) { return blitz::Array<T,1>(blitz::all(i, dim)); }

/**
 * Common methods
 */
template <typename T, int N> static void bind_common (tp::array<T,N>& array) {
  array.object()->def("sum", &sum<T,N>, "Summation");
  array.object()->def("product", &product<T,N>, "Product");
}

/**
 * Non-complex methods
 */
template <typename T, int N> static void bind (tp::array<T,N>& array) {
  bind_common(array); 
  array.object()->def("mean", &mean<T,N>, "Arithmetic mean");
  array.object()->def("min", &min<T,N>, "Minimum value");
  array.object()->def("max", &max<T,N>, "Maximum value");
  array.object()->def("minIndex", &minIndex<T,N>, "Index of the minimum value (returns tuple.");
  array.object()->def("maxIndex", &maxIndex<T,N>, "Index of the maximum value (returns tuple.");
  array.object()->def("any", &any<T,N>, "True if the array is True anywhere.");
  array.object()->def("all", &all<T,N>, "True if the array is True everywhere.");
  array.object()->def("count", &count<T,N>, "Counts the number of times the expression is true anywhere.");
}

void bind_array_reduction () {
  bind(tp::bool_1);
  bind(tp::bool_2);
  bind(tp::bool_3);
  bind(tp::bool_4);
  
  bind(tp::int8_1);
  bind(tp::int8_2);
  bind(tp::int8_3);
  bind(tp::int8_4);
  
  bind(tp::int16_1);
  bind(tp::int16_2);
  bind(tp::int16_3);
  bind(tp::int16_4);
  
  bind(tp::int32_1);
  bind(tp::int32_2);
  bind(tp::int32_3);
  bind(tp::int32_4);
  
  bind(tp::int64_1);
  bind(tp::int64_2);
  bind(tp::int64_3);
  bind(tp::int64_4);
  
  bind(tp::uint8_1);
  bind(tp::uint8_2);
  bind(tp::uint8_3);
  bind(tp::uint8_4);
  
  bind(tp::uint16_1);
  bind(tp::uint16_2);
  bind(tp::uint16_3);
  bind(tp::uint16_4);
  
  bind(tp::uint32_1);
  bind(tp::uint32_2);
  bind(tp::uint32_3);
  bind(tp::uint32_4);
  
  bind(tp::uint64_1);
  bind(tp::uint64_2);
  bind(tp::uint64_3);
  bind(tp::uint64_4);
  
  bind(tp::float32_1);
  bind(tp::float32_2);
  bind(tp::float32_3);
  bind(tp::float32_4);
  
  bind(tp::float64_1);
  bind(tp::float64_2);
  bind(tp::float64_3);
  bind(tp::float64_4);
  
  //bind(tp::float128_1);
  //bind(tp::float128_2);
  //bind(tp::float128_3);
  //bind(tp::float128_4);
  
  bind_common(tp::complex64_1);
  bind_common(tp::complex64_2);
  bind_common(tp::complex64_3);
  bind_common(tp::complex64_4);
  
  bind_common(tp::complex128_1);
  bind_common(tp::complex128_2);
  bind_common(tp::complex128_3);
  bind_common(tp::complex128_4);
  
  //bind(tp::complex256_1);
  //bind(tp::complex256_2);
  //bind(tp::complex256_3);
  //bind(tp::complex256_4);
}
