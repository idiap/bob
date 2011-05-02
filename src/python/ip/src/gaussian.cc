/**
 * @file src/python/ip/src/gaussian.cc
 * @author <a href="mailto:Nikls.Johansson@idiap.ch">Niklas Johansson</a> 
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds Gaussian smoothing to python
 */

#include <boost/python.hpp>
#include <vector>
#include "ip/Gaussian.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* gaussiandoc = "Performs gaussian smoothing";

template<typename T, int N>
static void gaussian_apply(ip::Gaussian& self, const blitz::Array<T,N>& src, blitz::Array<T,N>& dst) {
	self(src, dst);
}


void bind_ip_gaussian() {
	class_<ip::Gaussian, boost::shared_ptr<ip::Gaussian> >("Gaussian", gaussiandoc, init<optional<const int, const int, const double> >((arg("radius_x")=1, arg("radius_y")=1, arg("sigma")=5.), "Create a gaussian smoother"))
		.def("__call__",  &gaussian_apply<uint8_t, 2>, (arg("self"), arg("src"), arg("dst")), "Smooth an image")
		.def("__call__",  &gaussian_apply<uint16_t, 2>, (arg("self"), arg("src"), arg("dst")), "Smooth an image")
		.def("__call__",  &gaussian_apply<double, 2>, (arg("self"), arg("src"), arg("dst")), "Smooth an image")
		;
}

