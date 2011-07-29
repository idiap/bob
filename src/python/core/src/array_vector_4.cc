/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sat  9 Jul 08:08:04 2011 CEST
 *
 * @brief Support for vectors of arrays
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
