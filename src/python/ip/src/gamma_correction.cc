/**
 * @file src/python/ip/src/gamma_correction.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds gamma correction into python 
 */


#include "core/python/ndarray.h"
#include "ip/gammaCorrection.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ip = Torch::ip;
namespace ca = Torch::core::array;

template <typename T, int N>
static void inner_gammaCorrection (tp::const_ndarray src, tp::ndarray dst,
    double g) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  ip::gammaCorrection<T>(src.bz<T,N>(), dst_, g);
}

static void gammaCorrection (tp::const_ndarray src, tp::ndarray dst, double g) {
  const ca::typeinfo& info = src.type();

  if (info.nd != 2) PYTHON_ERROR(TypeError, "gamma correction does not support input of type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_gammaCorrection<uint8_t,2>(src, dst, g);
    case ca::t_uint16:
      return inner_gammaCorrection<uint16_t,2>(src, dst, g);
    case ca::t_float64:
      return inner_gammaCorrection<double,2>(src, dst, g);
    default:
      PYTHON_ERROR(TypeError, "gamma correction does not support type '%s'", info.str().c_str());
  }
}

void bind_ip_gamma_correction() {
  def("gammaCorrection", &gammaCorrection, (arg("src"), arg("dst"), arg("gamma")), "Perform a power-law gamma correction on a 2D blitz array/image.");
}
