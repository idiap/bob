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

void bind_ip_spatiotempgrad() {
}
