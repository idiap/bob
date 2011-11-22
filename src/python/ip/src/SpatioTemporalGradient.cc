/**
 * @file python/ip/src/SpatioTemporalGradient.cc
 * @date Tue Sep 6 17:29:53 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Bindings to Spatio Temporal gradients
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "ip/SpatioTemporalGradient.h"
#include "core/cast.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tp = Torch::python;
namespace ca = Torch::core::array;
namespace tc = Torch::core;

static tuple forward_gradient_1(const ip::ForwardGradient& g, 
    tp::const_ndarray i) {
  
  blitz::Range all = blitz::Range::all();
  blitz::Array<double,2> i1;
  blitz::Array<double,2> i2;
  
  const ca::typeinfo& info = i.type();

  switch (info.dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,3> i_ = i.bz<uint8_t,3>();
        i1.reference(tc::cast<double,uint8_t>(i_(0,all,all)));
        i2.reference(tc::cast<double,uint8_t>(i_(1,all,all)));
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,3> i_ = i.bz<double,3>();
        i1.reference(i_(0,all,all));
        i2.reference(i_(1,all,all));
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "forward gradient call does not support array with type '%s'", info.str().c_str());
  }

  tp::ndarray Ex(ca::t_float64, info.shape[0], info.shape[1]);
  tp::ndarray Ey(ca::t_float64, info.shape[0], info.shape[1]);
  tp::ndarray Et(ca::t_float64, info.shape[0], info.shape[1]);
  blitz::Array<double,2> Ex_ = Ex.bz<double,2>();
  blitz::Array<double,2> Ey_ = Ey.bz<double,2>();
  blitz::Array<double,2> Et_ = Et.bz<double,2>();

  g(i1, i2, Ex_, Ey_, Et_);

  return make_tuple(Ex.self(), Ey.self(), Et.self());
}

static void forward_gradient_2(const ip::ForwardGradient& g,
    tp::const_ndarray i1, tp::const_ndarray i2,
    tp::ndarray Ex, tp::ndarray Ey, tp::ndarray Et) {

  const ca::typeinfo& info = i1.type();

  blitz::Array<double,2> Ex_ = Ex.bz<double,2>();
  blitz::Array<double,2> Ey_ = Ey.bz<double,2>();
  blitz::Array<double,2> Et_ = Et.bz<double,2>();
  
  switch (info.dtype) {
    case ca::t_uint8:
      {
        g(tc::cast<double,uint8_t>(i1.bz<uint8_t,2>()), 
            tc::cast<double,uint8_t>(i2.bz<uint8_t,2>()), Ex_, Ey_, Et_);
      }
      break;
    case ca::t_float64:
      {
        g(i1.bz<double,2>(), i2.bz<double,2>(), Ex_, Ey_, Et_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "forward gradient call does not support array with type '%s'", info.str().c_str());
  }
}

static tuple forward_gradient_3(const ip::ForwardGradient& g,
    tp::const_ndarray i1, tp::const_ndarray i2) {
  const ca::typeinfo& info = i1.type();

  tp::ndarray Ex(ca::t_float64, info.shape[0], info.shape[1]);
  tp::ndarray Ey(ca::t_float64, info.shape[0], info.shape[1]);
  tp::ndarray Et(ca::t_float64, info.shape[0], info.shape[1]);
  forward_gradient_2(g, i1, i2, Ex, Ey, Et);

  return make_tuple(Ex.self(), Ey.self(), Et.self());
}

static tuple central_gradient_1(const ip::CentralGradient& g, 
    tp::const_ndarray i) {
  
  blitz::Range all = blitz::Range::all();
  blitz::Array<double,2> i1;
  blitz::Array<double,2> i2;
  blitz::Array<double,2> i3;
  
  const ca::typeinfo& info = i.type();

  switch (info.dtype) {
    case ca::t_uint8:
      {
        blitz::Array<uint8_t,3> i_ = i.bz<uint8_t,3>();
        i1.reference(tc::cast<double,uint8_t>(i_(0,all,all)));
        i2.reference(tc::cast<double,uint8_t>(i_(1,all,all)));
        i3.reference(tc::cast<double,uint8_t>(i_(2,all,all)));
      }
      break;
    case ca::t_float64:
      {
        blitz::Array<double,3> i_ = i.bz<double,3>();
        i1.reference(i_(0,all,all));
        i2.reference(i_(1,all,all));
        i3.reference(i_(1,all,all));
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "central gradient call does not support array with type '%s'", info.str().c_str());
  }

  tp::ndarray Ex(ca::t_float64, info.shape[0], info.shape[1]);
  tp::ndarray Ey(ca::t_float64, info.shape[0], info.shape[1]);
  tp::ndarray Et(ca::t_float64, info.shape[0], info.shape[1]);
  blitz::Array<double,2> Ex_ = Ex.bz<double,2>();
  blitz::Array<double,2> Ey_ = Ey.bz<double,2>();
  blitz::Array<double,2> Et_ = Et.bz<double,2>();

  g(i1, i2, i3, Ex_, Ey_, Et_);

  return make_tuple(Ex.self(), Ey.self(), Et.self());
}

static void central_gradient_2(const ip::CentralGradient& g,
    tp::const_ndarray i1, tp::const_ndarray i2, tp::const_ndarray i3,
    tp::ndarray Ex, tp::ndarray Ey, tp::ndarray Et) {

  const ca::typeinfo& info = i1.type();

  blitz::Array<double,2> Ex_ = Ex.bz<double,2>();
  blitz::Array<double,2> Ey_ = Ey.bz<double,2>();
  blitz::Array<double,2> Et_ = Et.bz<double,2>();
  
  switch (info.dtype) {
    case ca::t_uint8:
      {
        g(tc::cast<double,uint8_t>(i1.bz<uint8_t,2>()), 
            tc::cast<double,uint8_t>(i2.bz<uint8_t,2>()),
            tc::cast<double,uint8_t>(i3.bz<uint8_t,2>()), Ex_, Ey_, Et_);
      }
      break;
    case ca::t_float64:
      {
        g(i1.bz<double,2>(), i2.bz<double,2>(), i3.bz<double,2>(),
            Ex_, Ey_, Et_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "central gradient call does not support array with type '%s'", info.str().c_str());
  }
}

static tuple central_gradient_3(const ip::CentralGradient& g,
    tp::const_ndarray i1, tp::const_ndarray i2, tp::const_ndarray i3) {
  const ca::typeinfo& info = i1.type();

  tp::ndarray Ex(ca::t_float64, info.shape[0], info.shape[1]);
  tp::ndarray Ey(ca::t_float64, info.shape[0], info.shape[1]);
  tp::ndarray Et(ca::t_float64, info.shape[0], info.shape[1]);
  central_gradient_2(g, i1, i2, i3, Ex, Ey, Et);

  return make_tuple(Ex.self(), Ey.self(), Et.self());
}

void bind_ip_spatiotempgrad() {
  class_<ip::ForwardGradient>("ForwardGradient", "This class computes the spatio-temporal gradient using a 2-term approximation composed of 2 separable kernels (one for the diference term and another one for the averaging term).", init<const blitz::Array<double,1>&, const blitz::Array<double,1>&, const blitz::TinyVector<int,2>&>((arg("diff_kernel"), arg("avg_kernel"), arg("shape")), "Constructor. We initialize with the shape of the images we need to treat and with the kernels to be applied. The shape is used by the internal buffers.\n\n  diff_kernel\n    The kernel that contains the difference operation. Typically, this is [1; -1]. Note the kernel is mirrored during the convolution operation. To obtain a [-1; +1] sliding operator, specify [+1; -1]. This kernel must have a size = 2.\n\n  avg_kernel\n    The kernel that contains the spatial averaging operation. This kernel is typically [+1; +1]. This kernel must have a size = 2.\n\n  shape\n    This is the shape of the images to be treated. This has to match the input image height x width specifications (in that order)."))
    .add_property("shape", make_function(&ip::ForwardGradient::getShape, return_value_policy<copy_const_reference>()), &ip::ForwardGradient::setShape, "The internal buffer shape")
    .add_property("diff_kernel", make_function(&ip::ForwardGradient::getDiffKernel, return_value_policy<copy_const_reference>()), &ip::ForwardGradient::setDiffKernel, "The difference kernel")
    .add_property("avg_kernel", make_function(&ip::ForwardGradient::getAvgKernel, return_value_policy<copy_const_reference>()), &ip::ForwardGradient::setAvgKernel, "The averaging kernel")
    .def("__call__", &forward_gradient_1, (arg("s")))
    .def("__call__", &forward_gradient_3, (arg("i1"), arg("i2")))
    .def("__call__", &forward_gradient_2, (arg("i1"), arg("i2"), arg("Ex"), arg("Ey"), arg("Et")))
    ;

  class_<ip::HornAndSchunckGradient, bases<ip::ForwardGradient> >("HornAndSchunckGradient", "This class computes the spatio-temporal gradient using the same approximation as the one described by Horn & Schunck in the paper titled 'Determining Optical Flow', published in 1981, Artificial Intelligence, * Vol. 17, No. 1-3, pp. 185-203.\n\nThis is equivalent to convolving the image sequence with the following (separate) kernels:\n\nEx = 1/4 * ([-1 +1]^T * ([+1 +1]*(i1)) + [-1 +1]^T * ([+1 +1]*(i2)))\n\nEy = 1/4 * ([+1 +1]^T * ([-1 +1]*(i1)) + [+1 +1]^T * ([-1 +1]*(i2)))\n\nEt = 1/4 * ([+1 +1]^T * ([+1 +1]*(i1)) - [+1 +1]^T * ([+1 +1]*(i2)))", init<const blitz::TinyVector<int,2>&>((arg("shape")), "We initialize with the shape of the images we need to treat. The shape is used by the internal buffers.\n\nThe difference kernel for this operator is [+1/4; -1/4]\n\nThe averaging kernel for this oeprator is [+1; +1]."))
    ;

  class_<ip::CentralGradient>("CentralGradient",  "This class computes the spatio-temporal gradient using a 3-term approximation composed of 2 separable kernels (one for the diference term and another one for the averaging term).", init<const blitz::Array<double,1>&, const blitz::Array<double,1>&, const blitz::TinyVector<int,2>&>((arg("diff_kernel"), arg("avg_kernel"), arg("shape")), "Constructor. We initialize with the shape of the images we need to treat and with the kernels to be applied. The shape is used by the internal buffers.\n\n  diff_kernel\n    The kernel that contains the difference operation. Typically, this is [1; 0; -1]. Note the kernel is mirrored during the convolution operation. To obtain a [-1; 0; +1] sliding operator, specify [+1; 0; -1]. This kernel must have a size = 3.\n\n  avg_kernel\n    The kernel that contains the spatial averaging operation. This kernel is typically [+1; +1; +1]. This kernel must have a size = 3.\n\n  shape\n    This is the shape of the images to be treated. This has to match the input image height x width specifications (in that order)."))
    .add_property("shape", make_function(&ip::CentralGradient::getShape, return_value_policy<copy_const_reference>()), &ip::CentralGradient::setShape, "The internal buffer shape")
    .add_property("diff_kernel", make_function(&ip::CentralGradient::getDiffKernel, return_value_policy<copy_const_reference>()), &ip::CentralGradient::setDiffKernel, "The difference kernel")
    .add_property("avg_kernel", make_function(&ip::CentralGradient::getAvgKernel, return_value_policy<copy_const_reference>()), &ip::CentralGradient::setAvgKernel, "The averaging kernel")
    .def("__call__", &central_gradient_1, (arg("s")))
    .def("__call__", &central_gradient_3, (arg("i1"), arg("i2"), arg("i3")))
    .def("__call__", &central_gradient_2, (arg("i1"), arg("i2"), arg("i3"), arg("Ex"), arg("Ey"), arg("Et")))
    ;
  
  class_<ip::SobelGradient, bases<ip::CentralGradient> >("SobelGradient", "This class computes the spatio-temporal gradient using a 3-D sobel filter. The gradients are calculated along the x, y and t directions. The Sobel operator can be decomposed into 2 1D kernels that are applied in sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 2 1] one can represent the operations like this:\n\nEx = h'(x)h(y)h(t)\n\nEy = h(x)h'(y)h(t)\n\nEt = h(x)h(y)h'(t)", init<const blitz::TinyVector<int,2>&>((arg("shape")), "We initialize with the shape of the images we need to treat. The shape is used by the internal buffers.\n\nThe difference kernel for this operator is [+1; 0; -1]\n\nThe averaging kernel for this oeprator is [+1; +2; +1]."))
    ;

  class_<ip::PrewittGradient, bases<ip::CentralGradient> >("PrewittGradient", "This class computes the spatio-temporal gradient using a 3-D sobel filter. The gradients are calculated along the x, y and t directions. It can be decomposed into 2 1D kernels that are applied in sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 1 1] one can represent the operations like this:\n\nEx = h'(x)h(y)h(t)\n\nEy = h(x)h'(y)h(t)\n\nEt = h(x)h(y)h'(t)", init<const blitz::TinyVector<int,2>&>((arg("shape")), "We initialize with the shape of the images we need to treat. The shape is used by the internal buffers.\n\nThe difference kernel for this operator is [+1; 0; -1]\n\nThe averaging kernel for this oeprator is [+1; +1; +1]."))
    ;

  class_<ip::IsotropicGradient, bases<ip::CentralGradient> >("IsotropicGradient", "This class computes the spatio-temporal gradient using a isotropic filter. The gradients are calculated along the x, y and t directions. The Sobel operator can be decomposed into 2 1D kernels that are applied in sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 sqrt(2) 1] one can represent the operations like this:\n\nEx = h'(x)h(y)h(t)\n\nEy = h(x)h'(y)h(t)\n\nEt = h(x)h(y)h'(t)", init<const blitz::TinyVector<int,2>&>((arg("shape")), "We initialize with the shape of the images we need to treat. The shape is used by the internal buffers.\n\nThe difference kernel for this operator is [+1; 0; -1]\n\nThe averaging kernel for this oeprator is [+1; +sqrt(2); +1]."))
    ;
}
