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

void bind_core_arrayvectors_3 () {
  tp::vector_no_compare<blitz::Array<bool,3> >("bool_3");
  tp::vector_no_compare<blitz::Array<int8_t,3> >("int8_3");
  tp::vector_no_compare<blitz::Array<int16_t,3> >("int16_3");
  tp::vector_no_compare<blitz::Array<int32_t,3> >("int32_3");
  tp::vector_no_compare<blitz::Array<int64_t,3> >("int64_3");
  tp::vector_no_compare<blitz::Array<uint8_t,3> >("uint8_3");
  tp::vector_no_compare<blitz::Array<uint16_t,3> >("uint16_3");
  tp::vector_no_compare<blitz::Array<uint32_t,3> >("uint32_3");
  tp::vector_no_compare<blitz::Array<uint64_t,3> >("uint64_3");
  tp::vector_no_compare<blitz::Array<float,3> >("float32_3");
  tp::vector_no_compare<blitz::Array<double,3> >("float64_3");
  tp::vector_no_compare<blitz::Array<std::complex<float>,3> >("complex64_3");
  tp::vector_no_compare<blitz::Array<std::complex<double>,3> >("complex128_3");
}
