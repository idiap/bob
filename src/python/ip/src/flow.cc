/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue  8 Mar 15:12:40 2011 
 *
 * @brief Binds a few Optical Flow methods to python
 */

#include <boost/python.hpp>

#include "ip/HornAndSchunckFlow.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* hsdoc = "Objects of this class, after configuration, can calculate the Optical Flow between two sequences of images (i1, the starting image and i2, the final image). It does this using the iterative method described by Horn & Schunck in the paper titled \"Determining Optical Flow\", published in 1981, Artificial Intelligence, Vol. 17, No. 1-3, pp. 185-203.";

void bind_ip_flow() {
  //Horn & Schunck operator
  class_<ip::HornAndSchunckFlow, boost::shared_ptr<ip::HornAndSchunckFlow> >("HornAndSchunckFlow", hsdoc, init<float, size_t>((arg("alpha"), arg("iterations")), "Constructs a new HornAndSchunckFlow estimator using a certain weight alpha and pre-programmed to perform a number of iterations."))
    .add_property("alpha", &ip::HornAndSchunckFlow::getAlpha, &ip::HornAndSchunckFlow::setAlpha)
    .add_property("iterations", &ip::HornAndSchunckFlow::getIterations, &ip::HornAndSchunckFlow::setIterations)
    .def("__call__", &ip::HornAndSchunckFlow::operator(), (arg("image1"), arg("image2"), arg("u"), arg("v")), "Call an object of this type to compute the flow. u and v should be initialized or set to zero (if we are to compute the flow from scratch).")
    ;
}
