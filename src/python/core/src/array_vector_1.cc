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
