/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 17 Nov 15:46:36 2011 CET
 *
 * @brief Binds the Multiscale Retinex algorith into python
 */

#include "core/python/ndarray.h"
#include "ip/MultiscaleRetinex.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

static const char* msr_doc = "Applies the Multiscale Retinex algorithm";

template <typename T> static void inner_call (ip::MultiscaleRetinex& obj, 
    tp::const_ndarray input, tp::ndarray output) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  obj(input.bz<T,2>(), output_);
}

static void py_call (ip::MultiscaleRetinex& obj, tp::const_ndarray input,
    tp::ndarray output) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: return inner_call<uint8_t>(obj, input, output);
    case ca::t_uint16: return inner_call<uint16_t>(obj, input, output);
    case ca::t_float64: return inner_call<double>(obj, input, output);
    default: PYTHON_ERROR(TypeError, "MultiscaleRetinex __call__ operator does not support array with type '%s'", info.str().c_str());
  }
}

void bind_ip_msr() {
	class_<ip::MultiscaleRetinex, boost::shared_ptr<ip::MultiscaleRetinex> >("MultiscaleRetinex", msr_doc, init<optional<const size_t, const int, const int, const double> >((arg("n_scales")=1,arg("size_min")=1, arg("size_step")=1, arg("sigma")=5.), "Creates a MultiscaleRetinex object."))
		.def("__call__", &py_call, (arg("self"), arg("src"), arg("dst")), "Applies the Multiscale Retinex algorithm to an image of type uint8, uint16 or double")
		;
}
