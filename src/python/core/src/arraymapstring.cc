/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sat 22 Jul 15:12:15 2011 CEST
 *
 * @brief Support for maps of arrays
 */

#include <complex>
#include <boost/format.hpp>
#include <blitz/array.h>

#include "core/array_type.h"
#include "core/python/map.h"

namespace tp = Torch::python;

template <typename T> static void bind_array_map(const char* fmt) {
  boost::format s(fmt);
# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) tp::map_no_compare<blitz::Array<T,D> >( (s % D).str().c_str() );
#include BOOST_PP_LOCAL_ITERATE()
}

void bind_core_arraymaps () {
  bind_array_map<float>("array_float_%d_map");
  bind_array_map<double>("array_double_%d_map");
}
