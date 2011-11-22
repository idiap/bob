/**
 * @file python/core/src/array_vector_4.cc
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

void bind_core_arrayvectors_4 () {
  tp::vector_no_compare<blitz::Array<bool,4> >("bool_4");
  tp::vector_no_compare<blitz::Array<int8_t,4> >("int8_4");
  tp::vector_no_compare<blitz::Array<int16_t,4> >("int16_4");
  tp::vector_no_compare<blitz::Array<int32_t,4> >("int32_4");
  tp::vector_no_compare<blitz::Array<int64_t,4> >("int64_4");
  tp::vector_no_compare<blitz::Array<uint8_t,4> >("uint8_4");
  tp::vector_no_compare<blitz::Array<uint16_t,4> >("uint16_4");
  tp::vector_no_compare<blitz::Array<uint32_t,4> >("uint32_4");
  tp::vector_no_compare<blitz::Array<uint64_t,4> >("uint64_4");
  tp::vector_no_compare<blitz::Array<float,4> >("float32_4");
  tp::vector_no_compare<blitz::Array<double,4> >("float64_4");
  tp::vector_no_compare<blitz::Array<std::complex<float>,4> >("complex64_4");
  tp::vector_no_compare<blitz::Array<std::complex<double>,4> >("complex128_4");
}
