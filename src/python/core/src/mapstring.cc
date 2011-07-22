/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Support for maps of scalars with std::string ids
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
