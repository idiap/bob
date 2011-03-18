/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue  8 Mar 15:12:40 2011 
 *
 * @brief Binds a few Optical Flow methods to python
 */

#include <boost/python.hpp>

#include "ip/HornAndSchunckFlow.h"

using namespace boost::python;
namespace of = Torch::ip::optflow;

static const char* hsdoc = "Calculates the Optical Flow between two sequences of images (i1, the starting image and i2, the final image). It does this using the iterative method described by Horn & Schunck in the paper titled \"Determining Optical Flow\", published in 1981, Artificial Intelligence, Vol. 17, No. 1-3, pp. 185-203. Parameters: i1 -- first frame, i2 -- second frame, (u,v) -- estimates of the speed in x,y directions (zero if uninitialized).";

void bind_ip_flow() {
  //Horn & Schunck 
  def("evalHornAndSchunckFlow", of::evalHornAndSchunckFlow, (arg("alpha"), arg("iterations"), arg("image1"), arg("image2"), arg("u"), arg("v")), hsdoc);
  def("evalHornAndSchunckEc2", &of::evalHornAndSchunckEc2, (arg("u"), arg("v"), arg("square_error")), "Calculates the square of the smoothness error (Ec^2) by using the formula described in the paper: Ec^2 = (u_bar - u)^2 + (v_bar - v)^2. Sets the input matrix with the discrete values.");
  def("evalHornAndSchunckEb", &of::evalHornAndSchunckEb, (arg("i1"), arg("i2"), arg("u"), arg("v"), arg("error")), "Calculates the brightness error (Eb) as defined in the paper: Eb = (Ex*u + Ey*v + Et). Sets the input matrix with the discrete values");
}
