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

static tuple forward_gradient_1d(const ip::ForwardGradient& g, 
    const blitz::Array<double,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = i(0,all,all);
  const blitz::Array<double,2> i2 = i(1,all,all);
  blitz::Array<double,2> Ex(i1.shape());
  blitz::Array<double,2> Ey(i1.shape());
  blitz::Array<double,2> Et(i1.shape());
  g(i1, i2, Ex, Ey, Et);
  return make_tuple(Ex, Ey, Et);
}

static tuple forward_gradient_2d(const ip::ForwardGradient& g,
    const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2) {
  blitz::Array<double,2> Ex(i1.shape());
  blitz::Array<double,2> Ey(i1.shape());
  blitz::Array<double,2> Et(i1.shape());
  g(i1, i2, Ex, Ey, Et);
  return make_tuple(Ex, Ey, Et);
}

static tuple forward_gradient_1i(const ip::ForwardGradient& g,
    const blitz::Array<uint8_t,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = tc::cast<double,uint8_t>(i(0,all,all));
  const blitz::Array<double,2> i2 = tc::cast<double,uint8_t>(i(1,all,all));
  blitz::Array<double,2> Ex(i1.shape());
  blitz::Array<double,2> Ey(i1.shape());
  blitz::Array<double,2> Et(i1.shape());
  g(i1, i2, Ex, Ey, Et);
  return make_tuple(Ex, Ey, Et);
}

static tuple forward_gradient_2i(const ip::ForwardGradient& g,
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2) {
  blitz::Array<double,2> Ex(i1.shape());
  blitz::Array<double,2> Ey(i1.shape());
  blitz::Array<double,2> Et(i1.shape());
  g(tc::cast<double,uint8_t>(i1), tc::cast<double,uint8_t>(i2), Ex, Ey, Et);
  return make_tuple(Ex, Ey, Et);
}

static tuple central_gradient_1d(const ip::CentralGradient& g,
    const blitz::Array<double,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = i(0,all,all);
  const blitz::Array<double,2> i2 = i(1,all,all);
  const blitz::Array<double,2> i3 = i(2,all,all);
  blitz::Array<double,2> Ex(i1.shape());
  blitz::Array<double,2> Ey(i1.shape());
  blitz::Array<double,2> Et(i1.shape());
  g(i1, i2, i3, Ex, Ey, Et);
  return make_tuple(Ex, Ey, Et);
}

static tuple central_gradient_2d(const ip::CentralGradient& g,
    const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2,
    const blitz::Array<double,2>& i3) {
  blitz::Array<double,2> Ex(i1.shape());
  blitz::Array<double,2> Ey(i1.shape());
  blitz::Array<double,2> Et(i1.shape());
  g(i1, i2, i3, Ex, Ey, Et);
  return make_tuple(Ex, Ey, Et);
}

static tuple central_gradient_1i(const ip::CentralGradient& g,
    const blitz::Array<uint8_t,3>& i) {
  blitz::Range all = blitz::Range::all();
  const blitz::Array<double,2> i1 = tc::cast<double,uint8_t>(i(0,all,all));
  const blitz::Array<double,2> i2 = tc::cast<double,uint8_t>(i(1,all,all));
  const blitz::Array<double,2> i3 = tc::cast<double,uint8_t>(i(2,all,all));
  blitz::Array<double,2> Ex(i1.shape());
  blitz::Array<double,2> Ey(i1.shape());
  blitz::Array<double,2> Et(i1.shape());
  g(i1, i2, i3, Ex, Ey, Et);
  return make_tuple(Ex, Ey, Et);
}

static tuple central_gradient_2i(const ip::CentralGradient& g, 
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
    const blitz::Array<uint8_t,2>& i3) {
  blitz::Array<double,2> Ex(i1.shape());
  blitz::Array<double,2> Ey(i1.shape());
  blitz::Array<double,2> Et(i1.shape());
  g(tc::cast<double,uint8_t>(i1), tc::cast<double,uint8_t>(i2),
      tc::cast<double,uint8_t>(i3), Ex, Ey, Et);
  return make_tuple(Ex, Ey, Et);
}

static blitz::Array<double, 2> laplacian_014(const blitz::Array<double,2>& i) {
  blitz::Array<double,2> o(i.shape());
  ip::laplacian_014(i, o);
  return o;
}

static blitz::Array<double, 2> laplacian_18(const blitz::Array<double,2>& i) {
  blitz::Array<double,2> o(i.shape());
  ip::laplacian_18(i, o);
  return o;
}

static blitz::Array<double, 2> laplacian_12(const blitz::Array<double,2>& i) {
  blitz::Array<double,2> o(i.shape());
  ip::laplacian_12(i, o);
  return o;
}

static const char laplacian_014_doc[] = "An approximation to the Laplacian operator. Using the following (non-separable) kernel:\n\n[ 0 -1  0]\n[-1  4 -1]\n[ 0 -1  0]\n\nThis is used as the Laplacian operator on OpenCV (multiplied by -1)";

static const char laplacian_18_doc[] = "An approximation to the Laplacian operator. Using the following (non-separable) kernel:\n\n[-1 -1 -1]\n[-1  8 -1]\n[-1 -1 -1]";

static const char laplacian_12_doc[] = "An approximation to the Laplacian operator. Using the following (non-separable) kernel:\n\n[-1 -2 -1]\n[-2 12 -2]\n[-1 -2 -1]\n\nThis is used on the Horn & Schunck paper (multiplied by -1/12)";

void bind_ip_spatiotempgrad() {
  class_<ip::ForwardGradient>("ForwardGradient", "This class computes the spatio-temporal gradient using a 2-term approximation composed of 2 separable kernels (one for the diference term and another one for the averaging term).", init<const blitz::Array<double,1>&, const blitz::Array<double,1>&, const blitz::TinyVector<int,2>&>((arg("diff_kernel"), arg("avg_kernel"), arg("shape")), "Constructor. We initialize with the shape of the images we need to treat and with the kernels to be applied. The shape is used by the internal buffers.\n\n  diff_kernel\n    The kernel that contains the difference operation. Typically, this is [1; -1]. Note the kernel is mirrored during the convolution operation. To obtain a [-1; +1] sliding operator, specify [+1; -1]. This kernel must have a size = 2.\n\n  avg_kernel\n    The kernel that contains the spatial averaging operation. This kernel is typically [+1; +1]. This kernel must have a size = 2.\n\n  shape\n    This is the shape of the images to be treated. This has to match the input image height x width specifications (in that order)."))
    .add_property("shape", make_function(&ip::ForwardGradient::getShape, return_value_policy<copy_const_reference>()), &ip::ForwardGradient::setShape, "The internal buffer shape")
    .add_property("diff_kernel", make_function(&ip::ForwardGradient::getDiffKernel, return_value_policy<copy_const_reference>()), &ip::ForwardGradient::setDiffKernel, "The difference kernel")
    .add_property("avg_kernel", make_function(&ip::ForwardGradient::getAvgKernel, return_value_policy<copy_const_reference>()), &ip::ForwardGradient::setAvgKernel, "The averaging kernel")
    .def("__call__", &ip::ForwardGradient::operator(), (arg("i1"), arg("i2"), 
        arg("u"), arg("v")))
    .def("__call__", &forward_gradient_1d, (arg("s")))
    .def("__call__", &forward_gradient_2d, (arg("i1"), arg("i2")))
    .def("__call__", &forward_gradient_1i, (arg("s")))
    .def("__call__", &forward_gradient_2i, (arg("i1"), arg("i2")))
    ;

  class_<ip::HornAndSchunckGradient, bases<ip::ForwardGradient> >("HornAndSchunckGradient", "This class computes the spatio-temporal gradient using the same approximation as the one described by Horn & Schunck in the paper titled 'Determining Optical Flow', published in 1981, Artificial Intelligence, * Vol. 17, No. 1-3, pp. 185-203.\n\nThis is equivalent to convolving the image sequence with the following (separate) kernels:\n\nEx = 1/4 * ([-1 +1]^T * ([+1 +1]*(i1)) + [-1 +1]^T * ([+1 +1]*(i2)))\n\nEy = 1/4 * ([+1 +1]^T * ([-1 +1]*(i1)) + [+1 +1]^T * ([-1 +1]*(i2)))\n\nEt = 1/4 * ([+1 +1]^T * ([+1 +1]*(i1)) - [+1 +1]^T * ([+1 +1]*(i2)))", init<const blitz::TinyVector<int,2>&>((arg("shape")), "We initialize with the shape of the images we need to treat. The shape is used by the internal buffers.\n\nThe difference kernel for this operator is [+1/4; -1/4]\n\nThe averaging kernel for this oeprator is [+1; +1]."))
    ;

  class_<ip::CentralGradient>("CentralGradient",  "This class computes the spatio-temporal gradient using a 3-term approximation composed of 2 separable kernels (one for the diference term and another one for the averaging term).", init<const blitz::Array<double,1>&, const blitz::Array<double,1>&, const blitz::TinyVector<int,2>&>((arg("diff_kernel"), arg("avg_kernel"), arg("shape")), "Constructor. We initialize with the shape of the images we need to treat and with the kernels to be applied. The shape is used by the internal buffers.\n\n  diff_kernel\n    The kernel that contains the difference operation. Typically, this is [1; 0; -1]. Note the kernel is mirrored during the convolution operation. To obtain a [-1; 0; +1] sliding operator, specify [+1; 0; -1]. This kernel must have a size = 3.\n\n  avg_kernel\n    The kernel that contains the spatial averaging operation. This kernel is typically [+1; +1; +1]. This kernel must have a size = 3.\n\n  shape\n    This is the shape of the images to be treated. This has to match the input image height x width specifications (in that order)."))
    .add_property("shape", make_function(&ip::CentralGradient::getShape, return_value_policy<copy_const_reference>()), &ip::CentralGradient::setShape, "The internal buffer shape")
    .add_property("diff_kernel", make_function(&ip::CentralGradient::getDiffKernel, return_value_policy<copy_const_reference>()), &ip::CentralGradient::setDiffKernel, "The difference kernel")
    .add_property("avg_kernel", make_function(&ip::CentralGradient::getAvgKernel, return_value_policy<copy_const_reference>()), &ip::CentralGradient::setAvgKernel, "The averaging kernel")
    .def("__call__", &ip::CentralGradient::operator(), (arg("i1"), arg("i2"),
        arg("i3"), arg("u"), arg("v")))
    .def("__call__", &central_gradient_1d, (arg("s")))
    .def("__call__", &central_gradient_2d, (arg("i1"), arg("i2"), arg("i3")))
    .def("__call__", &central_gradient_1i, (arg("s")))
    .def("__call__", &central_gradient_2i, (arg("i1"), arg("i2"), arg("i3")))
    ;
  
  class_<ip::SobelGradient, bases<ip::CentralGradient> >("SobelGradient", "This class computes the spatio-temporal gradient using a 3-D sobel filter. The gradients are calculated along the x, y and t directions. The Sobel operator can be decomposed into 2 1D kernels that are applied in sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 2 1] one can represent the operations like this:\n\nEx = h'(x)h(y)h(t)\n\nEy = h(x)h'(y)h(t)\n\nEt = h(x)h(y)h'(t)", init<const blitz::TinyVector<int,2>&>((arg("shape")), "We initialize with the shape of the images we need to treat. The shape is used by the internal buffers.\n\nThe difference kernel for this operator is [+1; 0; -1]\n\nThe averaging kernel for this oeprator is [+1; +2; +1]."))
    ;

  class_<ip::PrewittGradient, bases<ip::CentralGradient> >("PrewittGradient", "This class computes the spatio-temporal gradient using a 3-D sobel filter. The gradients are calculated along the x, y and t directions. It can be decomposed into 2 1D kernels that are applied in sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 1 1] one can represent the operations like this:\n\nEx = h'(x)h(y)h(t)\n\nEy = h(x)h'(y)h(t)\n\nEt = h(x)h(y)h'(t)", init<const blitz::TinyVector<int,2>&>((arg("shape")), "We initialize with the shape of the images we need to treat. The shape is used by the internal buffers.\n\nThe difference kernel for this operator is [+1; 0; -1]\n\nThe averaging kernel for this oeprator is [+1; +1; +1]."))
    ;

  class_<ip::IsotropicGradient, bases<ip::CentralGradient> >("IsotropicGradient", "This class computes the spatio-temporal gradient using a isotropic filter. The gradients are calculated along the x, y and t directions. The Sobel operator can be decomposed into 2 1D kernels that are applied in sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 sqrt(2) 1] one can represent the operations like this:\n\nEx = h'(x)h(y)h(t)\n\nEy = h(x)h'(y)h(t)\n\nEt = h(x)h(y)h'(t)", init<const blitz::TinyVector<int,2>&>((arg("shape")), "We initialize with the shape of the images we need to treat. The shape is used by the internal buffers.\n\nThe difference kernel for this operator is [+1; 0; -1]\n\nThe averaging kernel for this oeprator is [+1; +sqrt(2); +1]."))
    ;

  def("laplacian_014", &laplacian_014, (arg("input")), laplacian_014_doc);
  def("laplacian_014", &ip::laplacian_014, (arg("input"), arg("output")), laplacian_014_doc);
  def("laplacian_18", &laplacian_18, (arg("input")), laplacian_18_doc);
  def("laplacian_18", &ip::laplacian_18, (arg("input"), arg("output")), laplacian_18_doc);
  def("laplacian_12", &laplacian_12, (arg("input")), laplacian_12_doc);
  def("laplacian_12", &ip::laplacian_12, (arg("input"), arg("output")), laplacian_12_doc);
}
