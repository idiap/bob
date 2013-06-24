/**
 * @file ip/python/SIFT.cc
 * @date Thu Sep 13 12:11:00 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include <bob/ip/SIFT.h>

#include <boost/python/stl_iterator.hpp>

using namespace boost::python;

template <typename T>
static object inner_compute_descr_p(bob::ip::SIFT& op, 
  bob::python::const_ndarray src, boost::python::object kp) 
{
  // Converts keypoints python list into a std::vector of keypoints
  stl_input_iterator<boost::python::object> vkp(kp), dend;
  std::vector<boost::shared_ptr<bob::ip::GSSKeypoint> > vkp_ref;
  vkp_ref.reserve(len(kp));
  for(; vkp != dend; ++vkp) vkp_ref.push_back(extract<boost::shared_ptr<bob::ip::GSSKeypoint> >(*vkp));

  // Allocates array for the descriptors
  const blitz::TinyVector<int,3> sift_shape = op.getDescriptorShape();
  bob::python::ndarray dst(bob::core::array::t_float64, (int)len(kp), sift_shape(0), sift_shape(1), sift_shape(2));
  const blitz::Array<T,2> src_ = src.bz<T,2>();
  blitz::Array<double,4> dst_ = dst.bz<double,4>();
  op.computeDescriptor(src_, vkp_ref, dst_);

  return dst.self();
}

static object compute_descr_p(bob::ip::SIFT& op, 
  bob::python::const_ndarray src, boost::python::object kp) 
{
  const bob::core::array::typeinfo& info = src.type();
  
  if(info.nd == 2)
  {
    switch(info.dtype) 
    {
      case bob::core::array::t_uint8: return inner_compute_descr_p<uint8_t>(op, src, kp);
      case bob::core::array::t_uint16: return inner_compute_descr_p<uint16_t>(op, src, kp);
      case bob::core::array::t_float64: return inner_compute_descr_p<double>(op, src, kp);
      default:
        PYTHON_ERROR(TypeError, "bob.ip.SIFT.compute_descriptor() does not support array with type '%s'", info.str().c_str());
    }
  }
  else
    PYTHON_ERROR(TypeError, "bob.ip.SIFT.compute_descriptor() does not support array with " SIZE_T_FMT " dimensions", info.nd);
}

void bind_ip_sift() 
{
  class_<bob::ip::SIFT, boost::shared_ptr<bob::ip::SIFT> >("SIFT", "This class allows after configuration the extraction of SIFT descriptors.\n\nReference:\n'Distinctive Image Features from Scale-Invariant Keypoints', D. Lowe, International Journal of Computer Vision, 2004", init<const size_t, const size_t, const size_t, const size_t, const int, optional<const double, const double, const double, const double, const double, const double, const bob::sp::Extrapolation::BorderType> >((arg("height"), arg("width"), arg("n_octaves"), arg("n_scales"), arg("octave_min"), arg("sigma_n")=0.5, arg("sigma0")=1.6, arg("contrast_thres")=0.03, arg("edge_thres")=10., arg("norm_thres")=0.2, arg("kernel_radius_factor")=4., arg("border_type")=bob::sp::Extrapolation::Mirror), "Creates an object that allows the extraction of SIFT descriptors."))
      .def(init<bob::ip::SIFT&>(args("other")))
      .def(self == self)
      .def(self != self)
      .add_property("height", &bob::ip::SIFT::getHeight, &bob::ip::SIFT::setHeight, "The height of the images to process")
      .add_property("width", &bob::ip::SIFT::getWidth, &bob::ip::SIFT::setWidth, "The width of the images to process")
      .add_property("n_octaves", &bob::ip::SIFT::getNOctaves, &bob::ip::SIFT::setNOctaves, "The number of octaves of the pyramid")
      .add_property("n_intervals", &bob::ip::SIFT::getNIntervals, &bob::ip::SIFT::setNIntervals, "The number of intervals of the pyramid. Three additional scales will be computed in practice, as this is required for extracting SIFT features.")
      .add_property("octave_min", &bob::ip::SIFT::getOctaveMin, &bob::ip::SIFT::setOctaveMin, "The index of the minimum octave")
      .add_property("octave_max", &bob::ip::SIFT::getOctaveMax, "The index of the maximum octave (read-only). This is equal to octave_min+n_octaves-1.")
      .add_property("sigma_n", &bob::ip::SIFT::getSigmaN, &bob::ip::SIFT::setSigmaN, "The value sigma_n of the standard deviation for the nominal/initial octave/scale")
      .add_property("sigma0", &bob::ip::SIFT::getSigma0, &bob::ip::SIFT::setSigma0, "The value sigma0 of the standard deviation for the input image")
      .add_property("kernel_radius_factor", &bob::ip::SIFT::getKernelRadiusFactor, &bob::ip::SIFT::setKernelRadiusFactor, "Factor used to determine the kernel radii (size=2*radius+1). For each Gaussian kernel, the radius is equal to ceil(kernel_radius_factor*sigma_{octave,scale}).")
      .add_property("conv_border", &bob::ip::SIFT::getConvBorder, &bob::ip::SIFT::setConvBorder, "The way the extractor deals with convolution at the boundary of the image when computing the Gaussian scale space.")
      .add_property("contrast_threshold", &bob::ip::SIFT::getContrastThreshold, &bob::ip::SIFT::setContrastThreshold, "The contrast threshold used during keypoint detection")
      .add_property("edge_threshold", &bob::ip::SIFT::getEdgeThreshold, &bob::ip::SIFT::setEdgeThreshold, "The edge threshold used during keypoint detection")
      .add_property("norm_threshold", &bob::ip::SIFT::getNormThreshold, &bob::ip::SIFT::setNormThreshold, "The norm threshold used during descriptor normalization")
      .add_property("n_blocks", &bob::ip::SIFT::getNBlocks, &bob::ip::SIFT::setNBlocks, "The number of blocks for the descriptor")
      .add_property("n_bins", &bob::ip::SIFT::getNBins, &bob::ip::SIFT::setNBins, "The number of bins for the descriptor")
      .add_property("gaussian_window_size", &bob::ip::SIFT::getGaussianWindowSize, &bob::ip::SIFT::setGaussianWindowSize, "The Gaussian window size for the descriptor")
      .add_property("magnif", &bob::ip::SIFT::getMagnif, &bob::ip::SIFT::setMagnif, "The magnification factor for the descriptor")
      .add_property("norm_epsilon", &bob::ip::SIFT::getNormEpsilon, &bob::ip::SIFT::setNormEpsilon, "The epsilon value added during the descriptor normalization")
      .def("set_sigma0_no_init_smoothing", &bob::ip::SIFT::setSigma0NoInitSmoothing, (arg("self")), "Sets sigma0 such that there is not smoothing at the first scale of octave_min.")
      .def("compute_descriptor", &compute_descr_p, (arg("self"), arg("src"), arg("keypoints")), "Computes SIFT descriptor for a 2D/grayscale image, at the given keypoints. The dst array will be allocated and returned.")
      .def("get_descriptor_shape", &bob::ip::SIFT::getDescriptorShape, (arg("self")), "Returns the shape of a descriptor for a given keypoint")
    ;
}

