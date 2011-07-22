/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sat 22 Jul 15:12:15 2011 CEST
 *
 * @brief Support for maps of arrays with std::string keys
 */

#include <complex>
#include <boost/format.hpp>
#include <blitz/array.h>

#include "core/array_type.h"
#include "core/python/mapstring.h"

namespace tp = Torch::python;

template <typename T> static void bind_array_mapstring(const char* fmt) {
  boost::format s(fmt);
# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) tp::mapstring_no_compare<blitz::Array<T,D> >( (s % D).str().c_str() );
#include BOOST_PP_LOCAL_ITERATE()
}

void bind_core_arraymapstrings () {
  bind_array_mapstring<float>("array_float_%d_mapstring");
  bind_array_mapstring<double>("array_double_%d_mapstring");
}
