/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Support for maps of scalars 
 */

#include <complex>
#include "core/python/map.h"

namespace tp = Torch::python;

void bind_core_maps () {
//  tp::map<std::string>("string");
  tp::map<bool>("bool");
  tp::map<int8_t>("int8");
  tp::map<int16_t>("int16");
  tp::map<int32_t>("int32");
  tp::map<int64_t>("int64");
  tp::map<uint8_t>("uint8");
  tp::map<uint16_t>("uint16");
  tp::map<uint32_t>("uint32");
  tp::map<uint64_t>("uint64");
  tp::map<float>("float32");
  tp::map<double>("float64");
  tp::map<long double>("float128");
  tp::map<std::complex<float> >("complex64");
  tp::map<std::complex<double> >("complex128");
  tp::map<std::complex<long double> >("complex256");

# ifdef __APPLE__
  //for some unknown reason, on OSX we need to define the map<size_t>
  tp::map<size_t>("size");
# endif
}
