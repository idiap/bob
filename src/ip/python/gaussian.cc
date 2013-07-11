/**
 * @file ip/python/gaussian.cc
 * @date Wed Apr 27 10:25:00 2011 +0200
 * @author Niklas Johansson <niklas.johansson@idiap.ch>
 *
 * @brief Binds Gaussian smoothing to python
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <bob/python/ndarray.h>
#include <bob/ip/Gaussian.h>

using namespace boost::python;

static object py_getKernelY(const bob::ip::Gaussian& op) 
{
  const blitz::Array<double,1>& kernel = op.getKernelY();
  bob::python::ndarray kernel_new(bob::core::array::t_float64, 
    kernel.extent(0));
  blitz::Array<double,1> kernel_new_ = kernel_new.bz<double,1>();
  kernel_new_ = kernel;
  return kernel_new.self();
}

static object py_getKernelX(const bob::ip::Gaussian& op) 
{
  const blitz::Array<double,1>& kernel = op.getKernelX();
  bob::python::ndarray kernel_new(bob::core::array::t_float64, 
    kernel.extent(0));
  blitz::Array<double,1> kernel_new_ = kernel_new.bz<double,1>();
  kernel_new_ = kernel;
  return kernel_new.self();
}

template <typename T, int N>
static void inner_call_gs1(bob::ip::Gaussian& op, 
    bob::python::const_ndarray src, bob::python::ndarray dst) 
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  op(src.bz<T,N>(), dst_);
}

static void call_gs1(bob::ip::Gaussian& op, 
    bob::python::const_ndarray src, bob::python::ndarray dst) 
{
  const bob::core::array::typeinfo& info = src.type();
  
  switch(info.nd)
  {
    case 2: 
      {
        switch(info.dtype) {
          case bob::core::array::t_uint8: return inner_call_gs1<uint8_t,2>(op, src, dst);
          case bob::core::array::t_uint16: return inner_call_gs1<uint16_t,2>(op, src, dst);
          case bob::core::array::t_float64: return inner_call_gs1<double,2>(op, src, dst);
          default:
            PYTHON_ERROR(TypeError, "Gaussian __call__ does not support array with type '%s'", info.str().c_str());
        }
      }
      break;
    case 3:
      {
        switch(info.dtype) {
          case bob::core::array::t_uint8: return inner_call_gs1<uint8_t,3>(op, src, dst);
          case bob::core::array::t_uint16: return inner_call_gs1<uint16_t,3>(op, src, dst);
          case bob::core::array::t_float64: return inner_call_gs1<double,3>(op, src, dst);
          default:
            PYTHON_ERROR(TypeError, "Gaussian __call__ does not support array with type '%s'", info.str().c_str());
        }
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "Gaussian __call__ does not support array with " SIZE_T_FMT " dimensions", info.nd);
  }
}

template <typename T>
static object inner_call_gs2_2d(bob::ip::Gaussian& op, 
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
static object inner_call_gs2_3d(bob::ip::Gaussian& op, 
    bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
  bob::python::ndarray dst(bob::core::array::t_float64, info.shape[0], 
    info.shape[1], info.shape[2]);
  blitz::Array<double,3> dst_ = dst.bz<double,3>();
  op(src.bz<T,3>(), dst_);
  return dst.self();
}

static object call_gs2(bob::ip::Gaussian& op, 
    bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
   
  switch(info.nd)
  {
    case 2: 
      {
        switch(info.dtype) {
          case bob::core::array::t_uint8: return inner_call_gs2_2d<uint8_t>(op, src);
          case bob::core::array::t_uint16: return inner_call_gs2_2d<uint16_t>(op, src);
          case bob::core::array::t_float64: return inner_call_gs2_2d<double>(op, src);
          default:
            PYTHON_ERROR(TypeError, "Gaussian __call__ does not support array with type '%s'", info.str().c_str());
        }
      }
      break;
    case 3:
      {
        switch(info.dtype) {
          case bob::core::array::t_uint8: return inner_call_gs2_3d<uint8_t>(op, src);
          case bob::core::array::t_uint16: return inner_call_gs2_3d<uint16_t>(op, src);
          case bob::core::array::t_float64: return inner_call_gs2_3d<double>(op, src);
          default:
            PYTHON_ERROR(TypeError, "Gaussian __call__ does not support array with type '%s'", info.str().c_str());
        }
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "Gaussian __call__ does not support array with " SIZE_T_FMT " dimensions", info.nd);
  }
}


void bind_ip_gaussian() 
{
  static const char* gaussiandoc = "This class allows after configuration to perform gaussian smoothing.";

  class_<bob::ip::Gaussian, boost::shared_ptr<bob::ip::Gaussian> >("Gaussian", gaussiandoc, init<optional<const size_t, const size_t, const double, const double, const bob::sp::Extrapolation::BorderType> >((arg("self"), arg("radius_y")=1, arg("radius_x")=1, arg("sigma_y")=sqrt(2.5), arg("sigma_x")=sqrt(2.5), arg("conv_border")=bob::sp::Extrapolation::Mirror), "Creates a gaussian smoother."))
      .def(init<bob::ip::Gaussian&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .add_property("radius_y", &bob::ip::Gaussian::getRadiusY, &bob::ip::Gaussian::setRadiusY, "The radius of the Gaussian along the y-axis (size of the kernel=2*radius+1)")
      .add_property("radius_x", &bob::ip::Gaussian::getRadiusX, &bob::ip::Gaussian::setRadiusX, "The radius of the Gaussian along the x-axis (size of the kernel=2*radius+1)")
      .add_property("sigma_y", &bob::ip::Gaussian::getSigmaY, &bob::ip::Gaussian::setSigmaY, "The variance of the Gaussian along the y-axis")
      .add_property("sigma_x", &bob::ip::Gaussian::getSigmaX, &bob::ip::Gaussian::setSigmaX, "The variance of the Gaussian along the x-axis")
      .add_property("conv_border", &bob::ip::Gaussian::getConvBorder, &bob::ip::Gaussian::setConvBorder, "The extrapolation method used by the convolution at the border")
      .add_property("kernel_y", &py_getKernelY, "The values of the y-kernel (read only access)")
      .add_property("kernel_x", &py_getKernelX, "The values of the x-kernel (read only access)")
      .def("reset", &bob::ip::Gaussian::reset, (arg("self"), arg("radius_y")=1, arg("radius_x")=1, arg("sigma_y")=sqrt(2.5), arg("sigma_x")=sqrt(2.5), arg("conv_border")=bob::sp::Extrapolation::Mirror), "Resets the parametrization of the Gaussian")
      .def("__call__", &call_gs1, (arg("self"), arg("src"), arg("dst")), "Smoothes an image (2D/grayscale or color 3D/color). The dst array should have the expected type (numpy.float64) and the same size as the src array.")
      .def("__call__", &call_gs2, (arg("self"), arg("src")), "Smoothes an image (2D/grayscale or color 3D/color). The smoothed image is returned as a numpy array.")
    ;
}
