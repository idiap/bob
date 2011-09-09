/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue  8 Mar 15:12:40 2011 
 *
 * @brief Binds a few Optical Flow methods to python
 */

#include <boost/python.hpp>
#include "core/cast.h"
#include "ip/HornAndSchunckFlow.h"

using namespace boost::python;
namespace of = Torch::ip::optflow;
namespace tc = Torch::core;

tuple vanillahs_call_d(const of::VanillaHornAndSchunckFlow& f,
    double alpha, size_t iterations,
    const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  f(alpha, iterations, i1, i2, u, v);
  return make_tuple(u, v);
}

tuple vanillahs_call_u8(const of::VanillaHornAndSchunckFlow& f,
    double alpha, size_t iterations,
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  f(alpha, iterations, tc::cast<double,uint8_t>(i1), 
      tc::cast<double,uint8_t>(i2), u, v);
  return make_tuple(u, v);
}

void vanillahs_call_u8_2(const of::VanillaHornAndSchunckFlow& f,
    double alpha, size_t iterations,
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
    blitz::Array<double,2>& u, blitz::Array<double,2>& v) {
  f(alpha, iterations, tc::cast<double,uint8_t>(i1), 
      tc::cast<double,uint8_t>(i2), u, v);
}

blitz::Array<double,2> vanillahs_ec2(const of::VanillaHornAndSchunckFlow& f,
    const blitz::Array<double,2>& u, const blitz::Array<double,2>& v) {
  blitz::Array<double,2> error(u.shape());
  f.evalEc2(u, v, error);
  return error;
}

blitz::Array<double,2> vanillahs_eb_d(const of::VanillaHornAndSchunckFlow& f,
    const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2,
    const blitz::Array<double,2>& u, const blitz::Array<double,2>& v) {
  blitz::Array<double,2> error(u.shape());
  f.evalEb(i1, i2, u, v, error);
  return error;
}

blitz::Array<double,2> vanillahs_eb_u8(const of::VanillaHornAndSchunckFlow& f,
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
    const blitz::Array<double,2>& u, const blitz::Array<double,2>& v) {
  blitz::Array<double,2> error(u.shape());
  f.evalEb(tc::cast<double,uint8_t>(i1), tc::cast<double,uint8_t>(i2), u, v, 
      error);
  return error;
}

tuple hs_call_d(const of::HornAndSchunckFlow& f,
    double alpha, size_t iterations,
    const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2,
    const blitz::Array<double,2>& i3) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  f(alpha, iterations, i1, i2, i3, u, v);
  return make_tuple(u, v);
}

tuple hs_call_u8(const of::HornAndSchunckFlow& f,
    double alpha, size_t iterations,
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
    const blitz::Array<uint8_t,2>& i3) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  f(alpha, iterations, tc::cast<double,uint8_t>(i1), 
      tc::cast<double,uint8_t>(i2), tc::cast<double,uint8_t>(i3), u, v);
  return make_tuple(u, v);
}

void hs_call_u8_2(const of::HornAndSchunckFlow& f,
    double alpha, size_t iterations,
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
    const blitz::Array<uint8_t,2>& i3,
    blitz::Array<double,2>& u, blitz::Array<double,2>& v) {
  f(alpha, iterations, tc::cast<double,uint8_t>(i1), 
      tc::cast<double,uint8_t>(i2), tc::cast<double,uint8_t>(i3), u, v);
}

blitz::Array<double,2> hs_ec2(const of::HornAndSchunckFlow& f,
    const blitz::Array<double,2>& u, const blitz::Array<double,2>& v) {
  blitz::Array<double,2> error(u.shape());
  f.evalEc2(u, v, error);
  return error;
}

blitz::Array<double,2> hs_eb_d(const of::HornAndSchunckFlow& f,
    const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2,
    const blitz::Array<double,2>& i3, const blitz::Array<double,2>& u,
    const blitz::Array<double,2>& v) {
  blitz::Array<double,2> error(u.shape());
  f.evalEb(i1, i2, i3, u, v, error);
  return error;
}

blitz::Array<double,2> hs_eb_u8(const of::HornAndSchunckFlow& f,
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
    const blitz::Array<uint8_t,2>& i3, const blitz::Array<double,2>& u,
    const blitz::Array<double,2>& v) {
  blitz::Array<double,2> error(u.shape());
  f.evalEb(tc::cast<double,uint8_t>(i1), tc::cast<double,uint8_t>(i2), 
      tc::cast<double,uint8_t>(i3), u, v, error);
  return error;
}

blitz::Array<double,2> flow_error_d(const blitz::Array<double,2>& i1,
    const blitz::Array<double,2>& i2, const blitz::Array<double,2>& u,
    const blitz::Array<double,2>& v) {
  blitz::Array<double,2> error(u.shape());
  of::flowError(i1, i2, u, v, error);
  return error;
}

blitz::Array<double,2> flow_error_u8(const blitz::Array<uint8_t,2>& i1,
    const blitz::Array<uint8_t,2>& i2, const blitz::Array<double,2>& u,
    const blitz::Array<double,2>& v) {
  blitz::Array<double,2> error(u.shape());
  of::flowError(tc::cast<double,uint8_t>(i1), tc::cast<double,uint8_t>(i2), 
      u, v, error);
  return error;
}

void bind_ip_flow() {
  //Horn & Schunck 
  class_<of::VanillaHornAndSchunckFlow>("VanillaHornAndSchunckFlow", "Calculates the Optical Flow between two sequences of images (i1, the starting image and i2, the final image). It does this using the iterative method described by Horn & Schunck in the paper titled \"Determining Optical Flow\", published in 1981, Artificial Intelligence, Vol. 17, No. 1-3, pp. 185-203. Parameters: i1 -- first frame, i2 -- second frame, (u,v) -- estimates of the speed in x,y directions (zero if uninitialized)", init<const blitz::TinyVector<int,2>&>((arg("shape")), "Initializes the vanilla Horn&Schunck operator with the size of images to be fed"))
      .def("__call__", &of::VanillaHornAndSchunckFlow::operator(), (arg("self"), arg("alpha"), arg("iterations"), arg("image1"), arg("image2"), arg("u"), arg("v")))
      .def("__call__", &vanillahs_call_d, (arg("self"), arg("alpha"), arg("iterations"), arg("image1"), arg("image2")))
      .def("__call__", &vanillahs_call_u8, (arg("self"), arg("alpha"), arg("iterations"), arg("image1"), arg("image2")))
      .def("__call__", &vanillahs_call_u8_2, (arg("self"), arg("alpha"), arg("iterations"), arg("image1"), arg("image2"), arg("u"), arg("v")))
      .def("evalEc2", &of::VanillaHornAndSchunckFlow::evalEc2, (arg("self"), arg("u"), arg("v"), arg("square_error")), "Calculates the square of the smoothness error (Ec^2) by using the formula described in the paper: Ec^2 = (u_bar - u)^2 + (v_bar - v)^2. Sets the input matrix with the discrete values.")
      .def("evalEb", &of::VanillaHornAndSchunckFlow::evalEb, (arg("self"), arg("i1"), arg("i2"), arg("u"), arg("v"), arg("error")), "Calculates the brightness error (Eb) as defined in the paper: Eb = (Ex*u + Ey*v + Et). Sets the input matrix with the discrete values")
      .def("evalEc2", &vanillahs_ec2, (arg("self"), arg("u"), arg("v")), "Calculates the square of the smoothness error (Ec^2) by using the formula described in the paper: Ec^2 = (u_bar - u)^2 + (v_bar - v)^2. Sets the input matrix with the discrete values.")
      .def("evalEb", &vanillahs_eb_d, (arg("self"), arg("i1"), arg("i2"), arg("u"), arg("v")), "Calculates the brightness error (Eb) as defined in the paper: Eb = (Ex*u + Ey*v + Et). Sets the input matrix with the discrete values")
      .def("evalEb", &vanillahs_eb_u8, (arg("self"), arg("i1"), arg("i2"), arg("u"), arg("v")), "Calculates the brightness error (Eb) as defined in the paper: Eb = (Ex*u + Ey*v + Et). Sets the input matrix with the discrete values")
      ;

  class_<of::HornAndSchunckFlow>("HornAndSchunckFlow", "This is a clone of the Vanilla HornAndSchunck method that uses a Sobel gradient estimator instead of the forward estimator used by the classical method. The Laplacian operator is also replaced with a more common method.", init<const blitz::TinyVector<int,2>&>((arg("shape")), "Initializes the vanilla Horn&Schunck operator with the size of images to be fed"))
      .def("__call__", &of::HornAndSchunckFlow::operator(), (arg("self"), arg("alpha"), arg("iterations"), arg("image1"), arg("image2"), arg("image3"), arg("u"), arg("v")))
      .def("__call__", &hs_call_d, (arg("self"), arg("alpha"), arg("iterations"), arg("image1"), arg("image2"), arg("image3")))
      .def("__call__", &hs_call_u8, (arg("self"), arg("alpha"), arg("iterations"), arg("image1"), arg("image2"), arg("image3")))
      .def("__call__", &hs_call_u8_2, (arg("self"), arg("alpha"), arg("iterations"), arg("image1"), arg("image2"), arg("image3"), arg("u"), arg("v")))
      .def("evalEc2", &of::HornAndSchunckFlow::evalEc2, (arg("self"), arg("u"), arg("v"), arg("square_error")), "Calculates the square of the smoothness error (Ec^2) by using the formula described in the paper: Ec^2 = (u_bar - u)^2 + (v_bar - v)^2. Sets the input matrix with the discrete values.")
      .def("evalEb", &of::HornAndSchunckFlow::evalEb, (arg("self"), arg("i1"), arg("i2"), arg("i3"), arg("u"), arg("v"), arg("error")), "Calculates the brightness error (Eb) as defined in the paper: Eb = (Ex*u + Ey*v + Et). Sets the input matrix with the discrete values")
      .def("evalEc2", &hs_ec2, (arg("self"), arg("u"), arg("v")), "Calculates the square of the smoothness error (Ec^2) by using the formula described in the paper: Ec^2 = (u_bar - u)^2 + (v_bar - v)^2. Sets the input matrix with the discrete values.")
      .def("evalEb", &hs_eb_d, (arg("self"), arg("i1"), arg("i2"), arg("i3"), arg("u"), arg("v")), "Calculates the brightness error (Eb) as defined in the paper: Eb = (Ex*u + Ey*v + Et). Sets the input matrix with the discrete values")
      .def("evalEb", &hs_eb_u8, (arg("self"), arg("i1"), arg("i2"), arg("i3"), arg("u"), arg("v")), "Calculates the brightness error (Eb) as defined in the paper: Eb = (Ex*u + Ey*v + Et). Sets the input matrix with the discrete values")
      ;

  def("flowError", &of::flowError, (arg("i1"), arg("i2"), arg("u"), arg("v"), arg("error")), "Computes the generalized flow error: E = i2(x-u,y-v) - i1(x,y))");
  def("flowError", &flow_error_d, (arg("i1"), arg("i2"), arg("u"), arg("v")), "Computes the generalized flow error: E = i2(x-u,y-v) - i1(x,y))");
  def("flowError", &flow_error_u8, (arg("i1"), arg("i2"), arg("u"), arg("v")), "Computes the generalized flow error: E = i2(x-u,y-v) - i1(x,y))");
}
