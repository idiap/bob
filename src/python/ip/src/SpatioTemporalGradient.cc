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

tuple forward_gradient_1d(const ip::ForwardGradient& g, 
    const blitz::Array<double,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = i(0,all,all);
  const blitz::Array<double,2> i2 = i(1,all,all);
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  g(i1, i2, u, v);
  return make_tuple(u, v);
}

tuple forward_gradient_2d(const ip::ForwardGradient& g,
    const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  g(i1, i2, u, v);
  return make_tuple(u, v);
}

tuple forward_gradient_1i(const ip::ForwardGradient& g, 
    const blitz::Array<uint8_t,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = tc::cast<double,uint8_t>(i(0,all,all));
  const blitz::Array<double,2> i2 = tc::cast<double,uint8_t>(i(1,all,all));
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  g(i1, i2, u, v);
  return make_tuple(u, v);
}

tuple forward_gradient_2i(const ip::ForwardGradient& g,
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  g(tc::cast<double,uint8_t>(i1), tc::cast<double,uint8_t>(i2), u, v);
  return make_tuple(u, v);
}

tuple central_gradient_1d(const ip::CentralGradient& g,
    const blitz::Array<double,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = i(0,all,all);
  const blitz::Array<double,2> i2 = i(1,all,all);
  const blitz::Array<double,2> i3 = i(2,all,all);
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  g(i1, i2, i3, u, v);
  return make_tuple(u, v);
}

tuple central_gradient_2d(const ip::CentralGradient& g,
    const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2,
    const blitz::Array<double,2>& i3) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  g(i1, i2, i3, u, v);
  return make_tuple(u, v);
}

tuple central_gradient_1i(const ip::CentralGradient& g,
    const blitz::Array<uint8_t,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = tc::cast<double,uint8_t>(i(0,all,all));
  const blitz::Array<double,2> i2 = tc::cast<double,uint8_t>(i(1,all,all));
  const blitz::Array<double,2> i3 = tc::cast<double,uint8_t>(i(2,all,all));
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  g(i1, i2, i3, u, v);
  return make_tuple(u, v);
}

tuple central_gradient_2i(const ip::CentralGradient& g, 
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
    const blitz::Array<uint8_t,2>& i3) {
  blitz::Array<double,2> u(i1.shape());
  u = 0;
  blitz::Array<double,2> v(i1.shape());
  v = 0;
  g(tc::cast<double,uint8_t>(i1), tc::cast<double,uint8_t>(i2),
      tc::cast<double,uint8_t>(i3), u, v);
  return make_tuple(u, v);
}

static const char FORWARD_DOC[] = "This class can compute the spatio-temporal gradient using the same approximation as the one described by Horn & Schunck in the paper titled 'Determining Optical Flow', published in 1981, Artificial Intelligence, Vol. 17, No. 1-3, pp. 185-203.\n\nThis is equivalent to convolving the image sequence with the following (separate) kernels:\n\n u = 1 * ([-1 +1]^T([+1 +1]*(i1)) + [-1 +1]^T ([+1 +1]*(i2)))\n v = 1 * ([+1 +1]^T ([-1 +1]*(i1)) + [+1 +1]^T ([-1 +1]*(i2)))\n\nThis will make-up the following convoluted kernel:\n\n u = [ -1 -1 ]   [ -1 -1 ]\n     [ +1 +1 ] ; [ +1 +1 ]\n\n v = [ -1 +1 ]   [ -1 +1 ]\n     [ -1 +1 ] ; [ -1 +1 ]\n\nThis method returns the matrices u and v that indicate the movement intensity along the 'x' and 'y' directions respectively. The kernels are normalized with their L2 norm so they represent unitary transformations.";

static const char CENTRAL_DOC[] = "This class can compute the spatio-temporal gradient using a 3-D sobel filter. The gradients are only calculated along the 'x' and 'y' directions. The Sobel operator can be decomposed into 3 1D kernels that are applied in sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 2 1] one can represent the operations like this:\n\n                     [+1]             [1]\n u = h'(x)h(y)h(t) = [ 0] [1 2 1]  [2]\n                     [-1]        [1]\n\n                     [1]              [1]\n v = h(x)h'(y)h(t) = [2] [-1 0 +1]  [2]\n                     [1]          [1]\n\n The Sobel operator is an edge detector. It calculates the gradient direction in the center of the 3D structure shown above. The kernels are normalized with their L2 norm so they represent unitary transformations.";

void bind_ip_spatiotempgrad() {
  class_<ip::ForwardGradient>("ForwardGradient", FORWARD_DOC, init<const blitz::TinyVector<int,2>&>((arg("shape")), "Initializes the gradient operator with a shape that defines the internal buffer sizes. When using the operator, the given image sizes should match this shape"))
    .add_property("shape", make_function(&ip::ForwardGradient::getShape, return_value_policy<copy_const_reference>()), &ip::ForwardGradient::setShape, "The internal buffer shape")
    .def("__call__", &ip::ForwardGradient::operator(), (arg("i1"), arg("i2"), 
        arg("u"), arg("v")))
    .def("__call__", &forward_gradient_1d, (arg("s")))
    .def("__call__", &forward_gradient_2d, (arg("i1"), arg("i2")))
    .def("__call__", &forward_gradient_1i, (arg("s")))
    .def("__call__", &forward_gradient_2i, (arg("i1"), arg("i2")))
    ;

  class_<ip::CentralGradient>("CentralGradient", CENTRAL_DOC, init<const blitz::TinyVector<int,2>&>((arg("shape")), "Initializes the gradient operator with a shape that defines the internal buffer sizes. When using the operator, the given image sizes should match this shape"))
    .add_property("shape", make_function(&ip::ForwardGradient::getShape, return_value_policy<copy_const_reference>()), &ip::ForwardGradient::setShape, "The internal buffer shape")
    .def("__call__", &ip::CentralGradient::operator(), (arg("i1"), arg("i2"),
        arg("i3"), arg("u"), arg("v")))
    .def("__call__", &central_gradient_1d, (arg("s")))
    .def("__call__", &central_gradient_2d, (arg("i1"), arg("i2"), arg("i3")))
    .def("__call__", &central_gradient_1i, (arg("s")))
    .def("__call__", &central_gradient_2i, (arg("i1"), arg("i2"), arg("i3")))
    ;
}
