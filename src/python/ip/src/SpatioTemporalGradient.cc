/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 06 Sep 2011 16:59:25 CEST
 *
 * @brief Bindings to Spatio Temporal gradients
 */

#include <boost/python.hpp>
#include "ip/SpatioTemporalGradient.h"
#include "core/cast.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tc = Torch::core;

tuple forward_gradient_1d(const blitz::Array<double,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = i(0,all,all);
  const blitz::Array<double,2> i2 = i(1,all,all);
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  ip::ForwardGradient(i1, i2, u, v);
  return make_tuple(u, v);
}

tuple forward_gradient_2d(const blitz::Array<double,2>& i1,
      const blitz::Array<double,2>& i2) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  ip::ForwardGradient(i1, i2, u, v);
  return make_tuple(u, v);
}

tuple forward_gradient_1i(const blitz::Array<uint8_t,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = tc::cast<double,uint8_t>(i(0,all,all));
  const blitz::Array<double,2> i2 = tc::cast<double,uint8_t>(i(1,all,all));
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  ip::ForwardGradient(i1, i2, u, v);
  return make_tuple(u, v);
}

tuple forward_gradient_2i(const blitz::Array<uint8_t,2>& i1,
      const blitz::Array<uint8_t,2>& i2) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  ip::ForwardGradient(tc::cast<double,uint8_t>(i1),
      tc::cast<double,uint8_t>(i2), u, v);
  return make_tuple(u, v);
}

tuple central_gradient_1d(const blitz::Array<double,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = i(0,all,all);
  const blitz::Array<double,2> i2 = i(1,all,all);
  const blitz::Array<double,2> i3 = i(2,all,all);
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  ip::CentralGradient(i1, i2, i3, u, v);
  return make_tuple(u, v);
}

tuple central_gradient_2d(const blitz::Array<double,2>& i1,
      const blitz::Array<double,2>& i2, const blitz::Array<double,2>& i3) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  ip::CentralGradient(i1, i2, i3, u, v);
  return make_tuple(u, v);
}

tuple central_gradient_1i(const blitz::Array<uint8_t,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = tc::cast<double,uint8_t>(i(0,all,all));
  const blitz::Array<double,2> i2 = tc::cast<double,uint8_t>(i(1,all,all));
  const blitz::Array<double,2> i3 = tc::cast<double,uint8_t>(i(2,all,all));
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  ip::CentralGradient(i1, i2, i3, u, v);
  return make_tuple(u, v);
}

tuple central_gradient_2i(const blitz::Array<uint8_t,2>& i1,
      const blitz::Array<uint8_t,2>& i2, const blitz::Array<uint8_t,2>& i3) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  ip::CentralGradient(tc::cast<double,uint8_t>(i1), 
      tc::cast<double,uint8_t>(i2), tc::cast<double,uint8_t>(i3), u, v);
  return make_tuple(u, v);
}

static const char FORWARD_DOC[] = "This method computes the spatio-temporal gradient using the same approximation as the one described by Horn & Schunck in the paper titled 'Determining Optical Flow', published in 1981, Artificial Intelligence, Vol. 17, No. 1-3, pp. 185-203.\n\nThis is equivalent to convolving the image sequence with the following (separate) kernels:\n\n u = 1/4 * ([-1 +1]^T([+1 +1]*(i1)) + [-1 +1]^T ([+1 +1]*(i2)))\n v = 1/4 * ([+1 +1]^T ([-1 +1]*(i1)) + [+1 +1]^T ([-1 +1]*(i2)))\n\nThis will make-up the following convoluted kernel:\n\n u = [ -1/4 -1/4 ]   [ -1/4 -1/4 ]\n     [ +1/4 +1/4 ] ; [ +1/4 +1/4 ]\n\n v = [ -1/4 +1/4 ]   [ -1/4 +1/4 ]\n     [ -1/4 +1/4 ] ; [ -1/4 +1/4 ]\n\nThis method returns the matrices u and v that indicate the movement intensity along the 'x' and 'y' directions respectively.";

static const char CENTRAL_DOC[] = "This method computes the spatio-temporal gradient using a 3-D sobel filter. The gradients are only calculated along the 'x' and 'y' directions. The Sobel operator can be decomposed into 3 1D kernels that are applied in sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 2 1] one can represent the operations like this:\n\n                     [+1]             [1]\n u = h'(x)h(y)h(t) = [ 0] [1 2 1]  [2]\n                     [-1]        [1]\n\n                     [1]              [1]\n v = h(x)h'(y)h(t) = [2] [-1 0 +1]  [2]\n                     [1]          [1]\n\n The Sobel operator is an edge detector. It calculates the gradient direction in the center of the 3D structure shown above.";

void bind_ip_spatiotempgrad() {
  def("ForwardGradient", &forward_gradient_1d, (arg("s")), FORWARD_DOC);
  def("ForwardGradient", &forward_gradient_2d, (arg("i1"), arg("i2")),
      FORWARD_DOC);
  def("ForwardGradient", &forward_gradient_1i, (arg("s")), FORWARD_DOC);
  def("ForwardGradient", &forward_gradient_2i, (arg("i1"), arg("i2")),
      FORWARD_DOC);
  def("ForwardGradient_", &ip::ForwardGradient, (arg("i1"), arg("i2"), 
        arg("u"), arg("v")), FORWARD_DOC);
  def("CentralGradient", &central_gradient_1d, (arg("s")), CENTRAL_DOC);
  def("CentralGradient", &central_gradient_2d, (arg("i1"), arg("i2"),
        arg("i3")), CENTRAL_DOC);
  def("CentralGradient", &central_gradient_1i, (arg("s")), CENTRAL_DOC);
  def("CentralGradient", &central_gradient_2i, (arg("i1"), arg("i2"),
        arg("i3")), CENTRAL_DOC);
  def("CentralGradient_", &ip::CentralGradient, (arg("i1"), arg("i2"), 
        arg("i3"), arg("u"), arg("v")), FORWARD_DOC);
}
