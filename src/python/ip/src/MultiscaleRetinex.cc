/**
 * @file src/python/ip/src/MultiscaleRetinex.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds the Multiscale Retinex algorith into python
 */

#include <boost/python.hpp>
#include <vector>
#include "ip/MultiscaleRetinex.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* msr_doc = "Applies the Multiscale Retinex algorithm";

void bind_ip_msr() {
	class_<ip::MultiscaleRetinex, boost::shared_ptr<ip::MultiscaleRetinex> >("MultiscaleRetinex", msr_doc, init<optional<const size_t, const int, const int, const double> >((arg("n_scales")=1,arg("size_min")=1, arg("size_step")=1, arg("sigma")=5.), "Creates a MultiscaleRetinex object."))
		.def("__call__", (void (ip::MultiscaleRetinex::*)(const blitz::Array<uint8_t,2>&, blitz::Array<double,2>&))&ip::MultiscaleRetinex::operator()<uint8_t>, (arg("self"), arg("src"), arg("dst")), "Applies the Multiscale Retinex algorithm to an image")
		.def("__call__", (void (ip::MultiscaleRetinex::*)(const blitz::Array<uint16_t,2>&, blitz::Array<double,2>&))&ip::MultiscaleRetinex::operator()<uint16_t>, (arg("self"), arg("src"), arg("dst")), "Applies the Multiscale Retinex algorithm to an image")
		.def("__call__", (void (ip::MultiscaleRetinex::*)(const blitz::Array<double,2>&, blitz::Array<double,2>&))&ip::MultiscaleRetinex::operator()<double>, (arg("self"), arg("src"), arg("dst")), "Applies the Multiscale Retinex algorithm to an image")
		;
}

