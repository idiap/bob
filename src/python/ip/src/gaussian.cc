/**
 * @author Niklas Johansson <Niklas.Johansson@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds Gaussian smoothing to python
 */

#include "core/python/ndarray.h"
#include "ip/Gaussian.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ip = Torch::ip;
namespace ca = Torch::core::array;

template <typename T, int N>
static void inner_call_gs (ip::Gaussian& op, tp::const_ndarray src, tp::ndarray dst) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  op(src.bz<T,N>(), dst_);
}

static void call_gs (ip::Gaussian& op, tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  
  if (info.nd != 2) PYTHON_ERROR(TypeError, "gaussian smoothing does not support input of type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: return inner_call_gs<uint8_t,2>(op, src, dst);
    case ca::t_uint16: return inner_call_gs<uint16_t,2>(op, src, dst);
    case ca::t_float64: return inner_call_gs<double,2>(op, src, dst);
    default:
      PYTHON_ERROR(TypeError, "gaussian smoothing does not support type '%s'", info.str().c_str());
  }
}

void bind_ip_gaussian() {
  static const char* gaussiandoc = "Performs gaussian smoothing";

	class_<ip::Gaussian, boost::shared_ptr<ip::Gaussian> >("Gaussian", gaussiandoc, init<optional<const int, const int, const double, const double> >((arg("radius_y")=1, arg("radius_x")=1, arg("sigma_y")=5., arg("sigma_x")=5.), "Creates a gaussian smoother"))
		.def("__call__", &call_gs, (arg("self"), arg("src"), arg("dst")), "Smooth an image")
		;
}
