/**
 * @file python/core/src/arraymapstring.cc
 * @date Fri Jul 22 20:13:49 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Support for maps of arrays with std::string keys
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
