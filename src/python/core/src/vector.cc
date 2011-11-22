/**
 * @file python/core/src/vector.cc
 * @date Mon Apr 18 16:45:57 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Support for vectors of scalars
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
#include "core/python/vector.h"

namespace tp = Torch::python;

void bind_core_vectors () {
  tp::vector<std::string>("string");
  tp::vector<bool>("bool");
  tp::vector<int8_t>("int8");
  tp::vector<int16_t>("int16");
  tp::vector<int32_t>("int32");
  tp::vector<int64_t>("int64");
  tp::vector<uint8_t>("uint8");
  tp::vector<uint16_t>("uint16");
  tp::vector<uint32_t>("uint32");
  tp::vector<uint64_t>("uint64");
  tp::vector<float>("float32");
  tp::vector<double>("float64");
  tp::vector<long double>("float128");
  tp::vector<std::complex<float> >("complex64");
  tp::vector<std::complex<double> >("complex128");
  tp::vector<std::complex<long double> >("complex256");

# ifdef __APPLE__
  //for some unknown reason, on OSX we need to define the vector<size_t>
  tp::vector<size_t>("size");
# endif
}
