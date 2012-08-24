/**
 * @file python/ip/src/TanTriggs.cc
 * @date Fri Mar 18 18:09:25 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Tan and Triggs preprocessing filter to python
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "ip/TanTriggs.h"
#include "core/python/ndarray.h"

using namespace boost::python;

static const char* ttdoc = "Objects of this class, after configuration, can preprocess images. It does this using the method described by Tan and Triggs in the paper titled \" Enhanced_Local_Texture_Feature_Sets for_Face_Recognition_Under_Difficult_Lighting_Conditions\", published in 2007";

static object py_getKernel(const bob::ip::TanTriggs& op) 
{
  const blitz::Array<double,2>& kernel = op.getKernel();
  bob::python::ndarray kernel_new(bob::core::array::t_float64, 
    kernel.extent(0), kernel.extent(1));
  blitz::Array<double,2> kernel_new_ = kernel_new.bz<double,2>();
  kernel_new_ = kernel;
  return kernel_new.self();
}

template <typename T> 
static void inner_call1(bob::ip::TanTriggs& obj, 
  bob::python::const_ndarray src, bob::python::ndarray dst) 
{
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  obj(src.bz<T,2>(), dst_);
}

static void call1(bob::ip::TanTriggs& obj, bob::python::const_ndarray src,
  bob::python::ndarray dst) 
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_call1<uint8_t>(obj, src, dst);
    case bob::core::array::t_uint16:
      return inner_call1<uint16_t>(obj, src, dst);
    case bob::core::array::t_float64: 
      return inner_call1<double>(obj, src, dst);
    default: PYTHON_ERROR(TypeError, "TanTriggers __call__ does not support array with type '%s'", info.str().c_str());
  }
}

template <typename T>
static object inner_call2(bob::ip::TanTriggs& op, 
  bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
  bob::python::ndarray dst(bob::core::array::t_float64, info.shape[0], 
    info.shape[1]);
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<T,2>(), dst_);
  return dst.self();
}

static object call2(bob::ip::TanTriggs& op, bob::python::const_ndarray src)
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: return inner_call2<uint8_t>(op, src);
    case bob::core::array::t_uint16: return inner_call2<uint16_t>(op, src);
    case bob::core::array::t_float64: return inner_call2<double>(op, src);
    default:
      PYTHON_ERROR(TypeError, "TanTriggs __call__ does not support array with type '%s'", info.str().c_str());
  }
}

void bind_ip_tantriggs() {
  class_<bob::ip::TanTriggs, boost::shared_ptr<bob::ip::TanTriggs> >("TanTriggs", ttdoc, init<optional<const double, const double, const double, const int, const double, const double, const bob::sp::Extrapolation::BorderType> >((arg("gamma")=0.2, arg("sigma0")=1., arg("sigma1")=2., arg("radius")=2, arg("threshold")=10., arg("alpha")=0.1, arg("conv_border")=bob::sp::Extrapolation::Mirror), "Constructs a new Tan and Triggs filter."))
      .def(init<bob::ip::TanTriggs&>(args("other")))
      .def(self == self)
      .def(self != self)
      .add_property("gamma", &bob::ip::TanTriggs::getGamma, &bob::ip::TanTriggs::setGamma, "The value of gamma for the gamma correction")
      .add_property("sigma0", &bob::ip::TanTriggs::getSigma0, &bob::ip::TanTriggs::setSigma0, "The standard deviation of the inner Gaussian")
      .add_property("sigma1", &bob::ip::TanTriggs::getSigma1, &bob::ip::TanTriggs::setSigma1, "The standard deviation of the outer Gaussian")
      .add_property("radius", &bob::ip::TanTriggs::getRadius, &bob::ip::TanTriggs::setRadius, "The radius of the Difference of Gaussians filter along both axes (size of the kernel=2*radius+1)")
      .add_property("threshold", &bob::ip::TanTriggs::getThreshold, &bob::ip::TanTriggs::setThreshold, "The threshold used for the contrast equalization")
      .add_property("alpha", &bob::ip::TanTriggs::getAlpha, &bob::ip::TanTriggs::setAlpha, "The alpha value used for the contrast equalization")
      .add_property("conv_border", &bob::ip::TanTriggs::getConvBorder, &bob::ip::TanTriggs::setConvBorder, "The extrapolation method used by the convolution at the border")
      .add_property("kernel", &py_getKernel, "The values of the DoG filter (read only access)")
      .def("reset", &bob::ip::TanTriggs::reset, (arg("self"), arg("gamma")=0.2, arg("sigma0")=0.1, arg("sigma1")=0.2, arg("radius")=2, arg("threshold")=10., arg("alpha")=0.1, arg("conv_border")=bob::sp::Extrapolation::Mirror), "Resets the parametrization of the Tan and Triggs preprocessor")
      .def("__call__", &call1, (arg("self"), arg("src"), arg("dst")), "Preprocesses a 2D/grayscale image using the algorithm from Tan and Triggs. The dst array should have the expected type (numpy.float64) and the same size as the src array.")
      .def("__call__", &call2, (arg("self"), arg("src")), "Preprocesses a 2D/grayscale image using the algorithm from Tan and Triggs. The preprocessed image is returned as a 2D numpy array of type numpy.float64.")
    ;
}

