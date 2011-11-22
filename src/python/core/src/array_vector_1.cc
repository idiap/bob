/**
 * @file python/core/src/array_vector_1.cc
 * @date Fri Jul 29 22:22:48 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Support for vectors of arrays
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
#include <blitz/array.h>

#include "core/array_type.h"
#include "core/python/vector.h"

namespace tp = Torch::python;

void bind_core_arrayvectors_1 () {
  tp::vector_no_compare<blitz::Array<bool,1> >("bool_1");
  tp::vector_no_compare<blitz::Array<int8_t,1> >("int8_1");
  tp::vector_no_compare<blitz::Array<int16_t,1> >("int16_1");
  tp::vector_no_compare<blitz::Array<int32_t,1> >("int32_1");
  tp::vector_no_compare<blitz::Array<int64_t,1> >("int64_1");
  tp::vector_no_compare<blitz::Array<uint8_t,1> >("uint8_1");
  tp::vector_no_compare<blitz::Array<uint16_t,1> >("uint16_1");
  tp::vector_no_compare<blitz::Array<uint32_t,1> >("uint32_1");
  tp::vector_no_compare<blitz::Array<uint64_t,1> >("uint64_1");
  tp::vector_no_compare<blitz::Array<float,1> >("float32_1");
  tp::vector_no_compare<blitz::Array<double,1> >("float64_1");
  tp::vector_no_compare<blitz::Array<std::complex<float>,1> >("complex64_1");
  tp::vector_no_compare<blitz::Array<std::complex<double>,1> >("complex128_1");
}
