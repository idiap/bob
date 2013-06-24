/**
 * @file ip/python/MultiscaleRetinex.cc
 * @date Mon May 2 10:01:08 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Multiscale Retinex algorith into python
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

#include "bob/python/ndarray.h"
#include "bob/ip/MultiscaleRetinex.h"

using namespace boost::python;

template <typename T, int N> 
static void inner_call1(bob::ip::MultiscaleRetinex& op, 
    bob::python::const_ndarray src, bob::python::ndarray dst) 
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  op(src.bz<T,N>(), dst_);
}

static void py_call1(bob::ip::MultiscaleRetinex& op, bob::python::const_ndarray src,
    bob::python::ndarray dst) 
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.nd)
  {
    case 2:
      {
        switch (info.dtype) {
          case bob::core::array::t_uint8: return inner_call1<uint8_t,2>(op, src, dst);
          case bob::core::array::t_uint16: return inner_call1<uint16_t,2>(op, src, dst);
          case bob::core::array::t_float64: return inner_call1<double,2>(op, src, dst);
          default: PYTHON_ERROR(TypeError, "MultiscaleRetinex __call__ operator does not support array with type '%s'", info.str().c_str());
        }
      }
      break;
    case 3:
      {
        switch (info.dtype) {
          case bob::core::array::t_uint8: return inner_call1<uint8_t,3>(op, src, dst);
          case bob::core::array::t_uint16: return inner_call1<uint16_t,3>(op, src, dst);
          case bob::core::array::t_float64: return inner_call1<double,3>(op, src, dst);
          default: PYTHON_ERROR(TypeError, "MultiscaleRetinex __call__ operator does not support array with type '%s'", info.str().c_str());
        }
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "MultiscaleRetinex __call__ does not support array with " SIZE_T_FMT " dimensions", info.nd);
  }
}

template <typename T> 
static object inner_call2_2d(bob::ip::MultiscaleRetinex& op, 
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
static object inner_call2_3d(bob::ip::MultiscaleRetinex& op, 
    bob::python::const_ndarray src)
{
  const bob::core::array::typeinfo& info = src.type();
  bob::python::ndarray dst(bob::core::array::t_float64, info.shape[0], 
    info.shape[1], info.shape[3]);
  blitz::Array<double,3> dst_ = dst.bz<double,3>();
  op(src.bz<T,3>(), dst_);
  return dst.self();
}

static object py_call2(bob::ip::MultiscaleRetinex& op, 
    bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
   
  switch(info.nd)
  {
    case 2: 
      {
        switch (info.dtype) {
          case bob::core::array::t_uint8: return inner_call2_2d<uint8_t>(op, src);
          case bob::core::array::t_uint16: return inner_call2_2d<uint16_t>(op, src);
          case bob::core::array::t_float64: return inner_call2_2d<double>(op, src);
          default:
            PYTHON_ERROR(TypeError, "MultiscaleRetinex __call__ does not support array with type '%s'", info.str().c_str());
        }
      }
      break;
    case 3:
      {
        switch (info.dtype) {
          case bob::core::array::t_uint8: return inner_call2_3d<uint8_t>(op, src);
          case bob::core::array::t_uint16: return inner_call2_3d<uint16_t>(op, src);
          case bob::core::array::t_float64: return inner_call2_3d<double>(op, src);
          default:
            PYTHON_ERROR(TypeError, "MultiscaleRetinex __call__ does not support array with type '%s'", info.str().c_str());
        }
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "MultiscaleRetinex __call__ does not support array with " SIZE_T_FMT " dimensions", info.nd);
  }
}



void bind_ip_msr() {
  class_<bob::ip::MultiscaleRetinex, boost::shared_ptr<bob::ip::MultiscaleRetinex> >("MultiscaleRetinex", "This class allows after configuration to apply the Self Quotient Image algorithm to images.", init<optional<const size_t, const int, const int, const double, const bob::sp::Extrapolation::BorderType> >((arg("n_scales")=1,arg("size_min")=1, arg("size_step")=1, arg("sigma")=2., arg("conv_border")=bob::sp::Extrapolation::Mirror), "Creates a MultiscaleRetinex object."))
      .def(init<bob::ip::MultiscaleRetinex&>(args("other")))
      .def(self == self)
      .def(self != self)
      .add_property("n_scales", &bob::ip::MultiscaleRetinex::getNScales, &bob::ip::MultiscaleRetinex::setNScales, "The number of scales (Gaussian).")
      .add_property("size_min", &bob::ip::MultiscaleRetinex::getSizeMin, &bob::ip::MultiscaleRetinex::setSizeMin, "The radius (size=2*radius+1) of the kernel of the smallest Gaussian.")
      .add_property("size_step", &bob::ip::MultiscaleRetinex::getSizeStep, &bob::ip::MultiscaleRetinex::setSizeStep, "The step used to set the kernel size of other Gaussians (size_s=2*(size_min+s*size_step)+1).")
      .add_property("sigma", &bob::ip::MultiscaleRetinex::getSigma, &bob::ip::MultiscaleRetinex::setSigma, "The variance of the kernel of the smallest Gaussian (variance_s = sigma * (size_min+s*size_step)/size_min).")
      .add_property("conv_border", &bob::ip::MultiscaleRetinex::getConvBorder, &bob::ip::MultiscaleRetinex::setConvBorder, "The extrapolation method used by the convolution at the border")
      .def("reset", &bob::ip::MultiscaleRetinex::reset, (arg("self"), arg("n_scales")=1, arg("size_min")=1, arg("size_step")=1, arg("sigma")=2., arg("conv_border")=bob::sp::Extrapolation::Mirror), "Resets the parametrization of the MultiscaleRetinex object.")
      .def("__call__", &py_call1, (arg("self"), arg("src"), arg("dst")), "Applies the Self Quotient Image algorithm to an image (2D/grayscale or color 3D/color) of type uint8, uint16 or double. The dst array should have the type (numpy.float64) and the same size as the src array.")
      .def("__call__", &py_call2, (arg("self"), arg("src")), "Applies the Self Quotient Image algorithm to an image (2D/grayscale or color 3D/color) of type uint8, uint16 or double. The filtered image is returned as a numpy array.")
    ;
}
