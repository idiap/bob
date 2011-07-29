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

void bind_core_arrayvectors_2 () {
  tp::vector_no_compare<blitz::Array<bool,2> >("bool_2");
  tp::vector_no_compare<blitz::Array<int8_t,2> >("int8_2");
  tp::vector_no_compare<blitz::Array<int16_t,2> >("int16_2");
  tp::vector_no_compare<blitz::Array<int32_t,2> >("int32_2");
  tp::vector_no_compare<blitz::Array<int64_t,2> >("int64_2");
  tp::vector_no_compare<blitz::Array<uint8_t,2> >("uint8_2");
  tp::vector_no_compare<blitz::Array<uint16_t,2> >("uint16_2");
  tp::vector_no_compare<blitz::Array<uint32_t,2> >("uint32_2");
  tp::vector_no_compare<blitz::Array<uint64_t,2> >("uint64_2");
  tp::vector_no_compare<blitz::Array<float,2> >("float32_2");
  tp::vector_no_compare<blitz::Array<double,2> >("float64_2");
  tp::vector_no_compare<blitz::Array<std::complex<float>,2> >("complex64_2");
  tp::vector_no_compare<blitz::Array<std::complex<double>,2> >("complex128_2");
}
