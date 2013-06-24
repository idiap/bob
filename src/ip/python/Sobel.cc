/**
 * @file ip/python/Sobel.cc
 * @date Mon Aug 27 12:53:00 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds Sobel filter to python
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
#include "bob/ip/Sobel.h"

using namespace boost::python;

static object py_getKernelY(const bob::ip::Sobel& op) 
{
  const blitz::Array<double,2>& kernel = op.getKernelY();
  bob::python::ndarray kernel_new(bob::core::array::t_float64, 
    kernel.extent(0), kernel.extent(1));
  blitz::Array<double,2> kernel_new_ = kernel_new.bz<double,2>();
  kernel_new_ = kernel;
  return kernel_new.self();
}

static object py_getKernelX(const bob::ip::Sobel& op) 
{
  const blitz::Array<double,2>& kernel = op.getKernelX();
  bob::python::ndarray kernel_new(bob::core::array::t_float64, 
    kernel.extent(0), kernel.extent(1));
  blitz::Array<double,2> kernel_new_ = kernel_new.bz<double,2>();
  kernel_new_ = kernel;
  return kernel_new.self();
}

template <typename T>
static void inner_call_gs1(bob::ip::Sobel& op, 
  bob::python::const_ndarray src, bob::python::ndarray dst) 
{
  blitz::Array<T,3> dst_ = dst.bz<T,3>();
  op(src.bz<T,2>(), dst_);
}

static void call_gs1(bob::ip::Sobel& op, 
  bob::python::const_ndarray src, bob::python::ndarray dst) 
{
  const bob::core::array::typeinfo& info = src.type();
  
  if(info.dtype == bob::core::array::t_float64)
    inner_call_gs1<double>(op, src, dst);
  else
    PYTHON_ERROR(TypeError, "Sobel __call__ does not support array with type '%s' (only type double/float64)..", info.str().c_str());
}


void bind_ip_sobel() 
{
  static const char* sobeldoc = "This class allows after configuration to perform Sobel filtering.";

  class_<bob::ip::Sobel, boost::shared_ptr<bob::ip::Sobel> >("Sobel", sobeldoc, init<optional<const bool, const bool, const bob::sp::Conv::SizeOption, const bob::sp::Extrapolation::BorderType> >((arg("up_positive")=false, arg("left_positive")=false, arg("size_opt")=bob::sp::Conv::Same, arg("conv_border")=bob::sp::Extrapolation::Mirror), "Creates a Sobel filter."))
      .def(init<bob::ip::Sobel&>(args("other")))
      .def(self == self)
      .def(self != self)
      .add_property("up_positive", &bob::ip::Sobel::getUpPositive, &bob::ip::Sobel::setUpPositive, "Whether the upper part of the y-kernel filter is positive or not.")
      .add_property("left_positive", &bob::ip::Sobel::getLeftPositive, &bob::ip::Sobel::setLeftPositive, "Whether the left part of the x-kernel filter is positive or not.")
      .add_property("size_option", &bob::ip::Sobel::getSizeOption, &bob::ip::Sobel::setSizeOption, "The part of the output to keep when performing convolutions.")
      .add_property("conv_border", &bob::ip::Sobel::getConvBorder, &bob::ip::Sobel::setConvBorder, "The extrapolation method used by the convolution at the border")
      .add_property("kernel_y", &py_getKernelY, "The values of the y-kernel (read only access)")
      .add_property("kernel_x", &py_getKernelX, "The values of the x-kernel (read only access)")
      .def("__call__", &call_gs1, (arg("self"), arg("src"), arg("dst")), "Filter a 2D/grayscale image with the parametrized Sobel filter. The dst array should have the expected type (numpy.float64) and the expected size, which depends on the the size option.")
    ;
}
