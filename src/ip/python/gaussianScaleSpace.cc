/**
 * @file python/ip/src/gaussianScaleSpace.cc
 * @date Thu Sp 4 15:40:00 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds GaussianScaleSpace to python
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

#include <bob/core/python/ndarray.h>
#include <bob/ip/GaussianScaleSpace.h>

using namespace boost::python;

static object allocate_output(const bob::ip::GaussianScaleSpace& op)
{
  std::vector<blitz::Array<double,3> > dst;
  boost::python::list dst_p;
  for (int i=op.getOctaveMin(); i<=op.getOctaveMax(); ++i)
  {
    const blitz::TinyVector<int,3> shape = op.getOutputShape(i);
    bob::python::ndarray dst_i(bob::core::array::t_float64, shape(0), shape(1), shape(2));
    dst_p.append(dst_i);
    dst.push_back(dst_i.bz<double,3>());
  }
  return object(dst_p);
}


template <typename T>
static object inner_call_c(const bob::ip::GaussianScaleSpace& op, 
  bob::python::const_ndarray src, boost::python::object dst) 
{
  std::vector<blitz::Array<double,3> > dst_c;
  for (int i=0; i<len(dst); ++i)
  {
    bob::python::ndarray dst_i = boost::python::extract<bob::python::ndarray>(dst[i]);
    dst_c.push_back(dst_i.bz<double,3>());
  }
  op(src.bz<T,2>(), dst_c);
  return dst;
}

static object call_c(bob::ip::GaussianScaleSpace& op, 
  bob::python::const_ndarray src, boost::python::object dst) 
{
  const bob::core::array::typeinfo& info = src.type();
  
  if (info.nd == 2)
  {
    switch (info.dtype) 
    {
      case bob::core::array::t_uint8: return inner_call_c<uint8_t>(op, src, dst);
      case bob::core::array::t_uint16: return inner_call_c<uint16_t>(op, src, dst);
      case bob::core::array::t_float64: return inner_call_c<double>(op, src, dst);
      default:
        PYTHON_ERROR(TypeError, "GaussianScaleSpace __call__ does not support array with type '%s'", info.str().c_str());
    }
  }
  else
    PYTHON_ERROR(TypeError, "Gaussian __call__ does not support array with " SIZE_T_FMT " dimensions", info.nd);
}



template <typename T>
static object inner_call_p(const bob::ip::GaussianScaleSpace& op, 
  bob::python::const_ndarray src) 
{
  std::vector<blitz::Array<double,3> > dst;
  boost::python::list dst_p;
  for (int i=op.getOctaveMin(); i<=op.getOctaveMax(); ++i)
  {
    const blitz::TinyVector<int,3> shape = op.getOutputShape(i);
    bob::python::ndarray dst_i(bob::core::array::t_float64, shape(0), shape(1), shape(2));
    dst_p.append(dst_i);
    dst.push_back(dst_i.bz<double,3>());
  }
  op(src.bz<T,2>(), dst);
  return object(dst_p);
}

static object call_p(const bob::ip::GaussianScaleSpace& op, 
  bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
  
  if (info.nd == 2)
  {
    switch(info.dtype) 
    {
      case bob::core::array::t_uint8: return inner_call_p<uint8_t>(op, src);
      case bob::core::array::t_uint16: return inner_call_p<uint16_t>(op, src);
      case bob::core::array::t_float64: return inner_call_p<double>(op, src);
      default:
        PYTHON_ERROR(TypeError, "GaussianScaleSpace __call__ does not support array with type '%s'", info.str().c_str());
    }
  }
  else
    PYTHON_ERROR(TypeError, "Gaussian __call__ does not support array with " SIZE_T_FMT " dimensions", info.nd);
}

void bind_ip_gaussian_scale_space() 
{
  class_<bob::ip::GaussianScaleSpace, boost::shared_ptr<bob::ip::GaussianScaleSpace> >("GaussianScaleSpace", "This class allows after configuration the generation of Gaussian Pyramids that can be used to extract SIFT features.\n\nReference:\n'Distinctive Image Features from Scale-Invariant Keypoints', D. Lowe, International Journal of Computer Vision, 2004", init<const size_t, const size_t, const size_t, const size_t, const int, optional<const double, const double, const double, const bob::sp::Extrapolation::BorderType> >((arg("height"), arg("width"), arg("n_octaves"), arg("n_scales"), arg("octave_min"), arg("sigma_n")=0.5, arg("sigma0")=1.6, arg("kernel_radius_factor")=4., arg("border_type")=bob::sp::Extrapolation::Mirror), "Creates an object that allows the construction of Gaussian pyramids."))
      .def(init<bob::ip::GaussianScaleSpace&>(args("other")))
      .def(self == self)
      .def(self != self)
      .add_property("height", &bob::ip::GaussianScaleSpace::getHeight, &bob::ip::GaussianScaleSpace::setHeight, "The height of the images to process")
      .add_property("width", &bob::ip::GaussianScaleSpace::getWidth, &bob::ip::GaussianScaleSpace::setWidth, "The width of the images to process")
      .add_property("n_octaves", &bob::ip::GaussianScaleSpace::getNOctaves, &bob::ip::GaussianScaleSpace::setNOctaves, "The number of octaves of the pyramid")
      .add_property("n_intervals", &bob::ip::GaussianScaleSpace::getNIntervals, &bob::ip::GaussianScaleSpace::setNIntervals, "The number of intervals of the pyramid. Three additional scales will be computed in practice, as this is required for extracting SIFT features.")
      .add_property("octave_min", &bob::ip::GaussianScaleSpace::getOctaveMin, &bob::ip::GaussianScaleSpace::setOctaveMin, "The index of the minimum octave")
      .add_property("octave_max", &bob::ip::GaussianScaleSpace::getOctaveMax, "The index of the maximum octave (read-only). This is equal to octave_min+n_octaves-1.")
      .add_property("sigma_n", &bob::ip::GaussianScaleSpace::getSigmaN, &bob::ip::GaussianScaleSpace::setSigmaN, "The value sigma_n of the standard deviation for the nominal/initial octave/scale")
      .add_property("sigma0", &bob::ip::GaussianScaleSpace::getSigma0, &bob::ip::GaussianScaleSpace::setSigma0, "The value sigma0 of the standard deviation for the image of the first octave and first scale")
      .add_property("kernel_radius_factor", &bob::ip::GaussianScaleSpace::getKernelRadiusFactor, &bob::ip::GaussianScaleSpace::setKernelRadiusFactor, "Factor used to determine the kernel radii (size=2*radius+1). For each Gaussian kernel, the radius is equal to ceil(kernel_radius_factor*sigma_{octave,scale}).")
      .add_property("conv_border", &bob::ip::GaussianScaleSpace::getConvBorder, &bob::ip::GaussianScaleSpace::setConvBorder, "The way to deal with convolutions at the image boundary.")
      .def("get_gaussian", &bob::ip::GaussianScaleSpace::getGaussian, (arg("self"), arg("index")), "Returns the Gaussian at index/interval i")
      .def("set_sigma0_no_init_smoothing", &bob::ip::GaussianScaleSpace::setSigma0NoInitSmoothing, (arg("self")), "Sets sigma0 such that there is not smoothing at the first scale of octave_min.")
      .def("allocate_output", &allocate_output, (arg("self")), "Allocates a python list of arrays for the Gaussian pyramid.")
      .def("__call__", &call_c, (arg("self"), arg("src"), arg("dst")), "Computes a Gaussian Pyramid for an input 2D image, and put the results in the output dst. The output should already be allocated and of the correct size (using the allocate_output() method).")
      .def("__call__", &call_p, (arg("self"), arg("src")), "Computes a Gaussian Pyramid for an input 2D image, and allocate and return the results.")
    ;
}
