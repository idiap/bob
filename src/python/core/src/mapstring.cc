/**
 * @file python/core/src/mapstring.cc
 * @date Fri Jul 22 20:13:49 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Support for maps of scalars with std::string ids
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
#include "core/python/mapstring.h"

namespace tp = Torch::python;

void bind_core_mapstrings() {
//  tp::map<std::string>("string");
  tp::mapstring<bool>("bool");
  tp::mapstring<int8_t>("int8");
  tp::mapstring<int16_t>("int16");
  tp::mapstring<int32_t>("int32");
  tp::mapstring<int64_t>("int64");
  tp::mapstring<uint8_t>("uint8");
  tp::mapstring<uint16_t>("uint16");
  tp::mapstring<uint32_t>("uint32");
  tp::mapstring<uint64_t>("uint64");
  tp::mapstring<float>("float32");
  tp::mapstring<double>("float64");
  tp::mapstring<long double>("float128");
  tp::mapstring<std::complex<float> >("complex64");
  tp::mapstring<std::complex<double> >("complex128");
  tp::mapstring<std::complex<long double> >("complex256");

# ifdef __APPLE__
  //for some unknown reason, on OSX we need to define the mapstring<size_t>
  tp::mapstring<size_t>("size");
# endif
}
