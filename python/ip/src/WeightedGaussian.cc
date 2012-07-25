/**
 * @file python/ip/src/WeightedGaussian.cc
 * @date Thu Jul 19 18:42:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Binds weighted Gaussian smoothing to python (cf. Self Quotient Image)
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

#include "core/python/ndarray.h"
#include "ip/WeightedGaussian.h"

using namespace boost::python;

static object py_getUnweightedKernel(const bob::ip::WeightedGaussian& op) 
{
  const blitz::Array<double,2>& kernel = op.getUnweightedKernel();
  bob::python::ndarray kernel_new(bob::core::array::t_float64, 
    kernel.extent(0), kernel.extent(1));
  blitz::Array<double,2> kernel_new_ = kernel_new.bz<double,2>();
  kernel_new_ = kernel;
  return kernel_new.self();
}

template <typename T, int N>
static void inner_call_wgs1(bob::ip::WeightedGaussian& op, 
  bob::python::const_ndarray src, bob::python::ndarray dst) 
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  op(src.bz<T,N>(), dst_);
}

static void call_wgs1(bob::ip::WeightedGaussian& op, 
  bob::python::const_ndarray src, bob::python::ndarray dst) 
{
  const bob::core::array::typeinfo& info = src.type();
  
  switch(info.nd)
  {
    case 2: 
      {
        switch (info.dtype) {
          case bob::core::array::t_uint8: return inner_call_wgs1<uint8_t,2>(op, src, dst);
          case bob::core::array::t_uint16: return inner_call_wgs1<uint16_t,2>(op, src, dst);
          case bob::core::array::t_float64: return inner_call_wgs1<double,2>(op, src, dst);
          default:
            PYTHON_ERROR(TypeError, "WeightedGaussian __call__ does not support array of type '%s'.", info.str().c_str());
        }
      }
      break;
    case 3:
      {
        switch (info.dtype) {
          case bob::core::array::t_uint8: return inner_call_wgs1<uint8_t,3>(op, src, dst);
          case bob::core::array::t_uint16: return inner_call_wgs1<uint16_t,3>(op, src, dst);
          case bob::core::array::t_float64: return inner_call_wgs1<double,3>(op, src, dst);
          default:
            PYTHON_ERROR(TypeError, "WeightedGaussian __call__ does not support array of type '%s'.", info.str().c_str());
        }
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "WeightedGaussian __call__ does not support array with '%ld' dimensions.", info.nd);
  }    
}

template <typename T>
static object inner_call_wgs2_2d(bob::ip::WeightedGaussian& op, 
  bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
  bob::python::ndarray dst(bob::core::array::t_float64, info.shape[0], 
    info.shape[1]);
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<T,2>(), dst_);
  return dst.self();
}

template <typename T>
static object inner_call_wgs2_3d(bob::ip::WeightedGaussian& op, 
  bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
  bob::python::ndarray dst(bob::core::array::t_float64, info.shape[0], 
    info.shape[1], info.shape[2]);
  blitz::Array<double,3> dst_ = dst.bz<double,3>();
  op(src.bz<T,3>(), dst_);
  return dst.self();
}

static object call_wgs2(bob::ip::WeightedGaussian& op, 
  bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
   
  switch(info.nd)
  {
    case 2: 
      {
        switch (info.dtype) {
          case bob::core::array::t_uint8: return inner_call_wgs2_2d<uint8_t>(op, src);
          case bob::core::array::t_uint16: return inner_call_wgs2_2d<uint16_t>(op, src);
          case bob::core::array::t_float64: return inner_call_wgs2_2d<double>(op, src);
          default:
            PYTHON_ERROR(TypeError, "WeightedGaussian __call__ does not support array of type '%s'.", info.str().c_str());
        }
      }
      break;
    case 3:
      {
        switch (info.dtype) {
          case bob::core::array::t_uint8: return inner_call_wgs2_3d<uint8_t>(op, src);
          case bob::core::array::t_uint16: return inner_call_wgs2_3d<uint16_t>(op, src);
          case bob::core::array::t_float64: return inner_call_wgs2_3d<double>(op, src);
          default:
            PYTHON_ERROR(TypeError, "WeightedGaussian __call__ does not support array of type '%s'.", info.str().c_str());
        }
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "WeightedGaussian __call__ does not support array with '%ld' dimensions.", info.nd);
  }
}


void bind_ip_wgaussian() 
{
  static const char* wgaussiandoc = "This class performs weighted gaussian smoothing (anisotropic filtering). In particular, it is used by the Self Quotient Image (SQI) algorithm.";

	class_<bob::ip::WeightedGaussian, boost::shared_ptr<bob::ip::WeightedGaussian> >("WeightedGaussian", wgaussiandoc, init<optional<const size_t, const size_t, const double, const double, const enum bob::sp::Extrapolation::BorderType> >((arg("radius_y")=1, arg("radius_x")=1, arg("sigma2_y")=2., arg("sigma2_x")=2., arg("conv_border")=bob::sp::Extrapolation::Mirror), "Creates a weighted gaussian smoother."))
      .def(init<bob::ip::WeightedGaussian&>(args("other")))
      .def(self == self)
      .def(self != self)
      .add_property("radius_y", &bob::ip::WeightedGaussian::getRadiusY, &bob::ip::WeightedGaussian::setRadiusY, "The radius of the unweighted Gaussian along the y-axis (height of the kernel=2*radius_y+1)")
      .add_property("radius_x", &bob::ip::WeightedGaussian::getRadiusX, &bob::ip::WeightedGaussian::setRadiusX, "The radius of the unweighted Gaussian along the x-axis (width of the kernel=2*radius_x+1)")
      .add_property("sigma2_y", &bob::ip::WeightedGaussian::getSigma2Y, &bob::ip::WeightedGaussian::setSigma2Y, "The variance of the unweighted Gaussian along the y-axis")
      .add_property("sigma2_x", &bob::ip::WeightedGaussian::getSigma2X, &bob::ip::WeightedGaussian::setSigma2X, "The variance of the unweighted Gaussian along the x-axis")
      .add_property("conv_border", &bob::ip::WeightedGaussian::getConvBorder, &bob::ip::WeightedGaussian::setConvBorder, "The extrapolation method used by the convolution at the border")
      .add_property("unweighted_kernel", &py_getUnweightedKernel, "The values of the unweighted kernel (read only access)")
      .def("reset", &bob::ip::WeightedGaussian::reset, (arg("self"), arg("radius_y")=1, arg("radius_x")=1, arg("sigma2_y")=5., arg("sigma2_x")=5., arg("conv_border")=bob::sp::Extrapolation::Mirror), "Resets the parametrization of the Weighted Gaussian")
		  .def("__call__", &call_wgs1, (arg("self"), arg("src"), arg("dst")), "Smoothes an image (2D/grayscale or color 3D/color). The dst array should have the expected type (numpy.float64) and the same size as the src array.")
		  .def("__call__", &call_wgs2, (arg("self"), arg("src")), "Smoothes an image (2D/grayscale or color 3D/color). The smoothed image is returned as a numpy array.")
		;
}
