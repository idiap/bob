/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Support for vectors of scalars 
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
  tp::vector<size_t>("size");
}
