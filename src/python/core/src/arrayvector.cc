/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sat  9 Jul 08:08:04 2011 CEST
 *
 * @brief Support for vectors of arrays
 */

#include <complex>
#include <boost/format.hpp>
#include <blitz/array.h>

#include "core/array_type.h"
#include "core/python/vector.h"

namespace tp = Torch::python;

template <typename T> static void bind_array_vector(const char* fmt) {
  boost::format s(fmt);
# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) tp::vector_no_compare<blitz::Array<T,D> >( (s % D).str().c_str() );
#include BOOST_PP_LOCAL_ITERATE()
}

void bind_core_arrayvectors () {
  bind_array_vector<float>("array_float_%d_vector");
  bind_array_vector<double>("array_double_%d_vector");
}
