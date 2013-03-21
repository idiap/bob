/**
 * @file ip/python/HOG.cc
 * @date Wed Apr 18 18:35:48 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds HOG descriptors into python
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

#include <boost/python.hpp>
#include "bob/core/python/ndarray.h"
#include "bob/core/cast.h"
#include "bob/ip/HOG.h"

using namespace boost::python;

static void hog_compute_histogram__c(bob::python::const_ndarray mag, 
  bob::python::const_ndarray ori, bob::python::ndarray hist, 
  const bool init_hist=true, const bool full_orientation=false)
{
  const bob::core::array::typeinfo& infoMag = mag.type();
  const bob::core::array::typeinfo& infoOri = ori.type();
  const bob::core::array::typeinfo& infoHist = hist.type();

  if(infoMag.nd != 2 || infoOri.nd !=2)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram_() requires 2D input arrays.");
  if(infoHist.nd !=1)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram_() requires a 1D output array.");

  if(infoMag.dtype != bob::core::array::t_float64 || 
      infoOri.dtype != bob::core::array::t_float64 || 
      infoHist.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram_() requires input arrays of type \
       float64.");

  blitz::Array<double,1> hist_ = hist.bz<double,1>();
  bob::ip::hogComputeHistogram_(mag.bz<double,2>(), ori.bz<double,2>(), hist_,
    init_hist, full_orientation);
}

static object hog_compute_histogram__p(bob::python::const_ndarray mag, 
  bob::python::const_ndarray ori, const size_t nb_bins, 
  const bool full_orientation=false)
{
  const bob::core::array::typeinfo& infoMag = mag.type();
  const bob::core::array::typeinfo& infoOri = ori.type();

  if(infoMag.nd != 2 || infoOri.nd !=2)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram_() requires 2D input arrays.");

  if(infoMag.dtype != bob::core::array::t_float64 || 
      infoOri.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram_() requires input arrays of type \
       float64.");

  bob::python::ndarray hist(bob::core::array::t_float64, nb_bins);
  blitz::Array<double,1> hist_ = hist.bz<double,1>();
  bob::ip::hogComputeHistogram_(mag.bz<double,2>(), ori.bz<double,2>(), hist_,
    true, full_orientation);
  return hist.self();
}

static void hog_compute_histogram_c(bob::python::const_ndarray mag, 
  bob::python::const_ndarray ori, bob::python::ndarray hist, 
  const bool init_hist=true, const bool full_orientation=false)
{
  const bob::core::array::typeinfo& infoMag = mag.type();
  const bob::core::array::typeinfo& infoOri = ori.type();
  const bob::core::array::typeinfo& infoHist = hist.type();

  if(infoMag.nd != 2 || infoOri.nd !=2)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram() requires 2D input arrays.");
  if(infoHist.nd !=1)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram() requires a 1D output array.");

  if(infoMag.dtype != bob::core::array::t_float64 || 
      infoOri.dtype != bob::core::array::t_float64 || 
      infoHist.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram() requires input arrays of type \
       float64.");

  blitz::Array<double,1> hist_ = hist.bz<double,1>();
  bob::ip::hogComputeHistogram(mag.bz<double,2>(), ori.bz<double,2>(), 
    hist_, init_hist, full_orientation);
}

static object hog_compute_histogram_p(bob::python::const_ndarray mag, 
  bob::python::const_ndarray ori, const size_t nb_bins, 
  const bool full_orientation=false)
{
  const bob::core::array::typeinfo& infoMag = mag.type();
  const bob::core::array::typeinfo& infoOri = ori.type();

  if(infoMag.nd != 2 || infoOri.nd !=2)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram_() requires 2D input arrays.");

  if(infoMag.dtype != bob::core::array::t_float64 || 
      infoOri.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, 
      "bob.ip.hog_compute_histogram_() requires input arrays of type \
       float64.");

  bob::python::ndarray hist(bob::core::array::t_float64, nb_bins);
  blitz::Array<double,1> hist_ = hist.bz<double,1>();
  bob::ip::hogComputeHistogram(mag.bz<double,2>(), ori.bz<double,2>(), 
    hist_, true, full_orientation);
  return hist.self();
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hog_compute_histogram__c_overloads, 
  hog_compute_histogram__c, 3, 5)
BOOST_PYTHON_FUNCTION_OVERLOADS(hog_compute_histogram__p_overloads, 
  hog_compute_histogram__p, 3, 4)
BOOST_PYTHON_FUNCTION_OVERLOADS(hog_compute_histogram_c_overloads, 
  hog_compute_histogram_c, 3, 5)
BOOST_PYTHON_FUNCTION_OVERLOADS(hog_compute_histogram_p_overloads, 
  hog_compute_histogram_p, 3, 4)



template <int D> 
static void inner_normalize_block_(bob::python::const_ndarray hist, 
  bob::python::ndarray norm_hist, const bob::ip::BlockNorm block_norm, 
  const double eps, const double threshold)
{
  blitz::Array<double,1> norm_hist_ = norm_hist.bz<double,1>();
  bob::ip::normalizeBlock_(hist.bz<double,D>(), norm_hist_, block_norm, eps,
    threshold);
}

static void normalize_block__c(bob::python::const_ndarray hist, 
  bob::python::ndarray norm_hist, 
  const bob::ip::BlockNorm block_norm=bob::ip::L2, const double eps=1e-10, 
  const double threshold=0.2)
{
  const bob::core::array::typeinfo& infoHist = hist.type();
  const bob::core::array::typeinfo& infoNormHist = norm_hist.type();

  if(infoNormHist.nd !=1)
    PYTHON_ERROR(TypeError, 
      "bob.ip.normalize_block_() requires a 1D output array.");

  if(infoHist.dtype != bob::core::array::t_float64 || 
      infoNormHist.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, 
      "bob.ip.normalize_block_() requires input arrays of type float64.");

  switch(infoHist.nd)
  {
    case 1:
      inner_normalize_block_<1>(hist, norm_hist, block_norm, eps, threshold);
      break;
    case 2:
      inner_normalize_block_<2>(hist, norm_hist, block_norm, eps, threshold);
      break;
    case 3:
      inner_normalize_block_<3>(hist, norm_hist, block_norm, eps, threshold);
      break;
    default:
      PYTHON_ERROR(TypeError, 
        "bob.ip.normalize_block_() requires a 1D, 2D or 3D input array.");
  } 
}

static object normalize_block__p(bob::python::const_ndarray hist, 
  const bob::ip::BlockNorm block_norm=bob::ip::L2, const double eps=1e-10, 
  const double threshold=0.2)
{
  const bob::core::array::typeinfo& infoHist = hist.type();

  if(infoHist.dtype != bob::core::array::t_float64) 
    PYTHON_ERROR(TypeError, 
      "bob.ip.normalize_block_() requires input arrays of type float64.");

  switch(infoHist.nd)
  {
    case 1:
      {
        bob::python::ndarray norm_hist(bob::core::array::t_float64, 
          infoHist.shape[0]);
        inner_normalize_block_<1>(hist, norm_hist, block_norm, eps, 
          threshold);
        return norm_hist.self();
      }
    case 2:
      {
        bob::python::ndarray norm_hist(bob::core::array::t_float64, 
          infoHist.shape[0]*infoHist.shape[1]);
        inner_normalize_block_<2>(hist, norm_hist, block_norm, eps, 
          threshold);
        return norm_hist.self();
      }
    case 3:
      {
        bob::python::ndarray norm_hist(bob::core::array::t_float64, 
          infoHist.shape[0]*infoHist.shape[1]*infoHist.shape[2]);
        inner_normalize_block_<3>(hist, norm_hist, block_norm, eps, 
          threshold);
        return norm_hist.self();
      }
    default:
      PYTHON_ERROR(TypeError, 
        "bob.ip.normalize_block_() requires a 1D, 2D or 3D input array.");
  } 
}


template <int D> 
static void inner_normalize_block( bob::python::const_ndarray hist, 
  bob::python::ndarray norm_hist, const bob::ip::BlockNorm block_norm, 
  const double eps, const double threshold)
{
  blitz::Array<double,1> norm_hist_ = norm_hist.bz<double,1>();
  bob::ip::normalizeBlock(hist.bz<double,D>(), norm_hist_, block_norm, eps, 
    threshold);
}

static void normalize_block_c(bob::python::const_ndarray hist, 
  bob::python::ndarray norm_hist,
  const bob::ip::BlockNorm block_norm=bob::ip::L2, const double eps=1e-10,
  const double threshold=0.2)
{
  const bob::core::array::typeinfo& infoHist = hist.type();
  const bob::core::array::typeinfo& infoNormHist = norm_hist.type();

  if(infoNormHist.nd !=1)
    PYTHON_ERROR(TypeError, 
      "bob.ip.normalize_block() requires a 1D output array.");

  if(infoHist.dtype != bob::core::array::t_float64 || 
      infoNormHist.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, 
      "bob.ip.normalize_block() requires input arrays of type float64.");

  switch(infoHist.nd)
  {
    case 1:
      inner_normalize_block<1>(hist, norm_hist, block_norm, eps, threshold);
      break;
    case 2:
      inner_normalize_block<2>(hist, norm_hist, block_norm, eps, threshold);
      break;
    case 3:
      inner_normalize_block<3>(hist, norm_hist, block_norm, eps, threshold);
      break;
    default:
      PYTHON_ERROR(TypeError, 
        "bob.ip.normalize_block() requires a 1D, 2D or 3D input array.");
  } 
}

static object normalize_block_p(bob::python::const_ndarray hist, 
  const bob::ip::BlockNorm block_norm=bob::ip::L2, const double eps=1e-10, 
  const double threshold=0.2)
{
  const bob::core::array::typeinfo& infoHist = hist.type();

  if(infoHist.dtype != bob::core::array::t_float64) 
    PYTHON_ERROR(TypeError, 
      "bob.ip.normalize_block() requires input arrays of type float64.");

  switch(infoHist.nd)
  {
    case 1:
      {
        bob::python::ndarray norm_hist(bob::core::array::t_float64, 
          infoHist.shape[0]);
        inner_normalize_block<1>(hist, norm_hist, block_norm, eps, 
          threshold);
        return norm_hist.self();
      }
    case 2:
      {
        bob::python::ndarray norm_hist(bob::core::array::t_float64, 
          infoHist.shape[0]*infoHist.shape[1]);
        inner_normalize_block<2>(hist, norm_hist, block_norm, eps, 
          threshold);
        return norm_hist.self();
      }
    case 3:
      {
        bob::python::ndarray norm_hist(bob::core::array::t_float64, 
          infoHist.shape[0]*infoHist.shape[1]*infoHist.shape[2]);
        inner_normalize_block<3>(hist, norm_hist, block_norm, eps, 
          threshold);
        return norm_hist.self();
      }
    default:
      PYTHON_ERROR(TypeError, 
        "bob.ip.normalize_block() requires a 1D, 2D or 3D input array.");
  } 
}

BOOST_PYTHON_FUNCTION_OVERLOADS(normalize_block__c_overloads, 
  normalize_block__c, 2, 5) 
BOOST_PYTHON_FUNCTION_OVERLOADS(normalize_block__p_overloads, 
  normalize_block__p, 1, 4) 
BOOST_PYTHON_FUNCTION_OVERLOADS(normalize_block_c_overloads, 
  normalize_block_c, 2, 5) 
BOOST_PYTHON_FUNCTION_OVERLOADS(normalize_block_p_overloads, 
  normalize_block_p, 1, 4) 



template <typename T> 
static void inner_gradient_maps_call1(bob::ip::GradientMaps& obj, 
  bob::python::const_ndarray input, bob::python::ndarray magnitude, 
  bob::python::ndarray orientation)
{
  blitz::Array<double,2> magnitude_ = magnitude.bz<double,2>();
  blitz::Array<double,2> orientation_ = orientation.bz<double,2>();
  obj.forward(input.bz<T,2>(), magnitude_, orientation_);
}

static void gradient_maps_call1(bob::ip::GradientMaps& obj, 
  bob::python::const_ndarray input, bob::python::ndarray magnitude, 
  bob::python::ndarray orientation) 
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_gradient_maps_call1<uint8_t>(obj, input, magnitude, 
              orientation);
    case bob::core::array::t_uint16:
      return inner_gradient_maps_call1<uint16_t>(obj, input, magnitude, 
              orientation);
    case bob::core::array::t_float64: 
      return inner_gradient_maps_call1<double>(obj, input, magnitude, 
              orientation);
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.GradientMaps __call__ does not support array with type '%s'.",
        info.str().c_str());
  }
}

static tuple gradient_maps_call1_p(bob::ip::GradientMaps& obj, 
  bob::python::const_ndarray input)
{
  const bob::core::array::typeinfo& info = input.type();
  bob::python::ndarray magnitude(bob::core::array::t_float64, 
    info.shape[0], info.shape[1]);
  bob::python::ndarray orientation(bob::core::array::t_float64, 
    info.shape[0], info.shape[1]);

  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      inner_gradient_maps_call1<uint8_t>(obj, input, magnitude, orientation);
      break;
    case bob::core::array::t_uint16:
      inner_gradient_maps_call1<uint16_t>(obj, input, magnitude, orientation);
      break;
    case bob::core::array::t_float64: 
      inner_gradient_maps_call1<double>(obj, input, magnitude, orientation);
      break;
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.GradientMaps __call__ does not support array with type '%s'.",
        info.str().c_str());
  }
  return make_tuple(magnitude, orientation);
}

template <typename T> 
static void inner_gradient_maps_call2(bob::ip::GradientMaps& obj, 
  bob::python::const_ndarray input, bob::python::ndarray magnitude, 
  bob::python::ndarray orientation)
{
  blitz::Array<double,2> magnitude_ = magnitude.bz<double,2>();
  blitz::Array<double,2> orientation_ = orientation.bz<double,2>();
  obj.forward_(input.bz<T,2>(), magnitude_, orientation_);
}

static void gradient_maps_call2(bob::ip::GradientMaps& obj, 
  bob::python::const_ndarray input, bob::python::ndarray magnitude, 
  bob::python::ndarray orientation) 
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_gradient_maps_call2<uint8_t>(obj, input, magnitude, 
              orientation);
    case bob::core::array::t_uint16:
      return inner_gradient_maps_call2<uint16_t>(obj, input, magnitude, 
              orientation);
    case bob::core::array::t_float64: 
      return inner_gradient_maps_call2<double>(obj, input, magnitude, 
              orientation);
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.GradientMaps __call__ does not support array with type '%s'.",
        info.str().c_str());
  }
}

static tuple gradient_maps_call2_p(bob::ip::GradientMaps& obj, 
  bob::python::const_ndarray input)
{
  const bob::core::array::typeinfo& info = input.type();
  bob::python::ndarray magnitude(bob::core::array::t_float64, 
    info.shape[0], info.shape[1]);
  bob::python::ndarray orientation(bob::core::array::t_float64, 
    info.shape[0], info.shape[1]);

  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      inner_gradient_maps_call2<uint8_t>(obj, input, magnitude, orientation);
      break;
    case bob::core::array::t_uint16:
      inner_gradient_maps_call2<uint16_t>(obj, input, magnitude, orientation);
      break;
    case bob::core::array::t_float64: 
      inner_gradient_maps_call2<double>(obj, input, magnitude, orientation);
      break;
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.GradientMaps __call__ does not support array with type '%s'.",
        info.str().c_str());
  }
  return make_tuple(magnitude, orientation);
}



template <typename T> 
static void inner_hog_call1(bob::ip::HOG<double>& obj, 
  bob::python::const_ndarray input, bob::python::ndarray output)
{
  blitz::Array<double,3> output_ = output.bz<double,3>();
  obj.forward(input.bz<T,2>(), output_);
}

template <typename T> 
static void inner_hog_call1_cast(bob::ip::HOG<double>& obj, 
  bob::python::const_ndarray input, bob::python::ndarray output)
{
  blitz::Array<double,2> input_c = bob::core::array::cast<double>(input.bz<T,2>());
  blitz::Array<double,3> output_ = output.bz<double,3>();
  obj.forward_(input_c, output_);
}

static void hog_call1(bob::ip::HOG<double>& obj, 
  bob::python::const_ndarray input, bob::python::ndarray output) 
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_hog_call1_cast<uint8_t>(obj, input, output);
    case bob::core::array::t_uint16:
      return inner_hog_call1_cast<uint16_t>(obj, input, output);
    case bob::core::array::t_float64: 
      return inner_hog_call1<double>(obj, input, output);
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.HOG __call__ does not support array with type '%s'.", 
        info.str().c_str());
  }
}

static object hog_call1_p(bob::ip::HOG<double>& obj, 
  bob::python::const_ndarray input) 
{
  const bob::core::array::typeinfo& info = input.type();
  const blitz::TinyVector<int,3> shape = obj.getOutputShape();
  bob::python::ndarray output(bob::core::array::t_float64, 
    shape(0), shape(1), shape(2));

  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      inner_hog_call1_cast<uint8_t>(obj, input, output);
      break;
    case bob::core::array::t_uint16:
      inner_hog_call1_cast<uint16_t>(obj, input, output);
      break;
    case bob::core::array::t_float64: 
      inner_hog_call1<double>(obj, input, output);
      break;
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.HOG __call__ does not support array with type '%s'.", 
        info.str().c_str());
  }

  return output.self();
}

template <typename T> 
static void inner_hog_call2(bob::ip::HOG<double>& obj, 
  bob::python::const_ndarray input, bob::python::ndarray output)
{
  blitz::Array<T,3> output_ = output.bz<T,3>();
  obj.forward_(input.bz<T,2>(), output_);
}

template <typename T> 
static void inner_hog_call2_cast(bob::ip::HOG<double>& obj, 
  bob::python::const_ndarray input, bob::python::ndarray output)
{
  blitz::Array<double,2> input_c = bob::core::array::cast<double>(input.bz<T,2>());
  blitz::Array<double,3> output_ = output.bz<double,3>();
  obj.forward_(input_c, output_);
}

static void hog_call2(bob::ip::HOG<double>& obj, 
  bob::python::const_ndarray input, bob::python::ndarray output) 
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_hog_call2_cast<uint8_t>(obj, input, output);
    case bob::core::array::t_uint16:
      return inner_hog_call2_cast<uint16_t>(obj, input, output);
    case bob::core::array::t_float64: 
      return inner_hog_call2<double>(obj, input, output);
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.HOG __call__ does not support array with type '%s'.", 
        info.str().c_str());
  }
}

static object hog_call2_p(bob::ip::HOG<double>& obj, 
  bob::python::const_ndarray input) 
{
  const bob::core::array::typeinfo& info = input.type();
  const blitz::TinyVector<int,3> shape = obj.getOutputShape();
  bob::python::ndarray output(bob::core::array::t_float64, 
    shape(0), shape(1), shape(2));

  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      inner_hog_call2_cast<uint8_t>(obj, input, output);
      break;
    case bob::core::array::t_uint16:
      inner_hog_call2_cast<uint16_t>(obj, input, output);
      break;
    case bob::core::array::t_float64: 
      inner_hog_call2<double>(obj, input, output);
      break;
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.HOG __call__ does not support array with type '%s'.", 
        info.str().c_str());
  }

  return output.self();
}


void bind_ip_hog() 
{
  static const char* gradientmaps_doc = 
    "Objects of this class, after configuration, can extract gradient \
     magnitude and orientation maps.";
  static const char* hog_doc = 
    "Objects of this class, after configuration, can extract \
     Histogram of Gradients (HOG) descriptors.";

  boost::python::enum_<bob::ip::GradientMagnitudeType>("GradientMagnitudeType")
    .value("Magnitude", bob::ip::Magnitude)
    .value("MagnitudeSquare", bob::ip::MagnitudeSquare)
    .value("SqrtMagnitude", bob::ip::SqrtMagnitude);
 
  boost::python::enum_<bob::ip::BlockNorm>("BlockNorm")
    .value("L2", bob::ip::L2)
    .value("L2Hys", bob::ip::L2Hys)
    .value("L1", bob::ip::L1)
    .value("L1sqrt", bob::ip::L1sqrt)
    .value("None", bob::ip::None);

  def("hog_compute_histogram_", &hog_compute_histogram__c, 
    hog_compute_histogram__c_overloads((arg("mag"), arg("ori"), arg("hist"),
    arg("init_hist")=true, arg("full_orientation")=false), 
    "Computes an Histogram of Gradients for a given 'cell'. The inputs are \
    the gradient magnitudes and the orientations for each pixel of the cell. \
    This variant does NOT check the inputs."));
  def("hog_compute_histogram_", &hog_compute_histogram__p, 
    hog_compute_histogram__p_overloads((arg("mag"), arg("ori"), 
    arg("nb_bins"), arg("full_orientation")=false), 
    "Computes an Histogram of Gradients for a given 'cell'. The inputs are \
    the gradient magnitudes and the orientations for each pixel of the cell. \
    This variant does NOT check the inputs."));
  def("hog_compute_histogram", &hog_compute_histogram_c, 
    hog_compute_histogram_c_overloads((arg("mag"), arg("ori"), 
    arg("hist"), arg("init_hist")=true, arg("full_orientation")=false), 
    "Computes an Histogram of Gradients for a given 'cell'. The inputs are \
    the gradient magnitudes and the orientations for each pixel of the \
    cell."));
  def("hog_compute_histogram", &hog_compute_histogram_p, 
    hog_compute_histogram_p_overloads((arg("mag"), arg("ori"), 
    arg("nb_bins"), arg("full_orientation")=false), 
    "Computes an Histogram of Gradients for a given 'cell'. The inputs are \
    the gradient magnitudes and the orientations for each pixel of the \
    cell."));

  def("normalize_block_", &normalize_block__c, 
    normalize_block__c_overloads((arg("hist"), arg("norm_hist"), 
    arg("block_norm")=bob::ip::L2, arg("eps")=1e-10, arg("threshold")=0.2), 
    "normalizes a set of cells (Histogram of Gradients), and returns \
    the corresponding block descriptor. This variant does NOT check the \
    inputs."));
  def("normalize_block_", &normalize_block__p, 
    normalize_block__p_overloads((arg("hist"),
    arg("block_norm")=bob::ip::L2, arg("eps")=1e-10, arg("threshold")=0.2), 
    "normalizes a set of cells (Histogram of Gradients), and returns \
    the corresponding block descriptor. This variant does NOT check the \
    inputs."));
  def("normalize_block", &normalize_block_c, 
    normalize_block_c_overloads((arg("hist"), arg("norm_hist"), 
    arg("block_norm")=bob::ip::L2, arg("eps")=1e-10, arg("threshold")=0.2), 
    "normalizes a set of cells (Histogram of Gradients), and returns \
    the corresponding block descriptor."));
  def("normalize_block", &normalize_block_p, 
    normalize_block_p_overloads((arg("hist"),
    arg("block_norm")=bob::ip::L2, arg("eps")=1e-10, arg("threshold")=0.2), 
    "normalizes a set of cells (Histogram of Gradients), and returns \
    the corresponding block descriptor."));

  class_<bob::ip::GradientMaps, boost::shared_ptr<bob::ip::GradientMaps> >(
      "GradientMaps", 
      gradientmaps_doc, 
      init<const size_t, const size_t, 
        optional<const bob::ip::GradientMagnitudeType> >(
          (arg("height"), arg("width"), arg("mag_type")=bob::ip::Magnitude),
          "Constructs a new Gradient maps extractor."))
    .def(init<bob::ip::GradientMaps&>(args("other")))
    .def(self == self)
    .def(self != self)
    .add_property("height", &bob::ip::GradientMaps::getHeight, 
      &bob::ip::GradientMaps::setHeight,
      "Height of the input image to process.")
    .add_property("width", &bob::ip::GradientMaps::getWidth, 
      &bob::ip::GradientMaps::setWidth,
      "Width of the input image to process.")
    .add_property("magnitude_type", 
      &bob::ip::GradientMaps::getGradientMagnitudeType, 
      &bob::ip::GradientMaps::setGradientMagnitudeType,
      "Type of the magnitude to use for the returned maps.")
    .def("resize", &bob::ip::GradientMaps::resize, 
      (arg("height"), arg("width")))
    .def("__call__", &gradient_maps_call1, 
      (arg("input"), arg("magnitude"), arg("orientation")),
      "Extract the gradient magnitude and orientation maps.")
    .def("__call__", &gradient_maps_call1_p, (arg("input")),
      "Extract the gradient magnitude and orientation maps.")
    .def("forward", &gradient_maps_call1, 
      (arg("input"), arg("magnitude"), arg("orientation")),
      "Extract the gradient magnitude and orientation maps.")
    .def("forward", &gradient_maps_call1_p, (arg("input")),
      "Extract the gradient magnitude and orientation maps.")
    .def("forward_", &gradient_maps_call2, 
      (arg("input"), arg("magnitude"), arg("orientation")),
      "Extract the gradient magnitude and orientation maps. This variant \
      does not check the inputs.")
    .def("forward_", &gradient_maps_call2_p, (arg("input")),
      "Extract the gradient magnitude and orientation maps. This variant \
      does not check the inputs.")
    ;

  class_<bob::ip::HOG<double>, boost::shared_ptr<bob::ip::HOG<double> > >(
      "HOG", 
      hog_doc, 
      init<const size_t, const size_t, 
        optional<const size_t, const bool, const size_t, const size_t, 
          const size_t, const size_t, const size_t, const size_t, 
          const size_t, const size_t> >(
        (arg("height"), arg("width"), arg("nb_bins")=8, 
         arg("full_orientation")=false, arg("cell_y")=4, arg("cell_x")=4, 
         arg("cell_ov_y")=0, arg("cell_ov_x")=0, arg("block_y")=4, 
         arg("block_x")=4, arg("block_ov_y")=0, arg("block_ov_x")=0),
        "Constructs a new HOG extractor."))
    .def(init<bob::ip::HOG<double>&>(args("other")))
    .def(self == self)
    .def(self != self)
    .add_property("height", &bob::ip::HOG<double>::getHeight,
      &bob::ip::HOG<double>::setHeight,
      "Height of the input image to process.")
    .add_property("width", &bob::ip::HOG<double>::getWidth,
      &bob::ip::HOG<double>::setWidth,
      "Width of the input image to process.")
    .add_property("magnitude_type", 
      &bob::ip::HOG<double>::getGradientMagnitudeType, 
      &bob::ip::HOG<double>::setGradientMagnitudeType,
      "Type of the magnitude to consider for the descriptors.")
    .add_property("cell_dim", &bob::ip::HOG<double>::getCellDim,
      &bob::ip::HOG<double>::setCellDim,
      "Dimensionality of a cell descriptor (i.e. the number of bins).")
    .add_property("full_orientation",
      &bob::ip::HOG<double>::getFullOrientation,
      &bob::ip::HOG<double>::setFullOrientation,
      "Whether the range [0,360] is used or not ([0,180] otherwise).")
    .add_property("cell_y", &bob::ip::HOG<double>::getCellHeight,
      &bob::ip::HOG<double>::setCellHeight,
      "Height of a cell.")
    .add_property("cell_x", &bob::ip::HOG<double>::getCellWidth,
      &bob::ip::HOG<double>::setCellWidth,
      "Width of a cell.")
    .add_property("cell_ov_y", &bob::ip::HOG<double>::getCellOverlapHeight,
      &bob::ip::HOG<double>::setCellOverlapHeight,
      "y-overlap between cells.")
    .add_property("cell_ov_x", &bob::ip::HOG<double>::getCellOverlapWidth,
      &bob::ip::HOG<double>::setCellOverlapWidth,
      "x-overlap between cells.")
    .add_property("block_y", &bob::ip::HOG<double>::getBlockHeight,
      &bob::ip::HOG<double>::setBlockHeight,
      "Height of a block (in terms of cells).")
    .add_property("block_x", &bob::ip::HOG<double>::getBlockWidth,
      &bob::ip::HOG<double>::setBlockWidth,
      "Width of a block (in terms of cells).")
    .add_property("block_ov_y", &bob::ip::HOG<double>::getBlockOverlapHeight,
      &bob::ip::HOG<double>::setBlockOverlapHeight,
      "y-overlap between blocks (in terms of cells).")
    .add_property("block_ov_x", &bob::ip::HOG<double>::getBlockOverlapWidth,
      &bob::ip::HOG<double>::setBlockOverlapWidth,
      "x-overlap between blocks (in terms of cells).")
    .add_property("block_norm", &bob::ip::HOG<double>::getBlockNorm, 
      &bob::ip::HOG<double>::setBlockNorm,
      "The type of norm used for normalizing blocks.")
    .add_property("block_norm_eps", &bob::ip::HOG<double>::getBlockNormEps, 
      &bob::ip::HOG<double>::setBlockNormEps,
      "Epsilon value used to avoid division by zeros when normalizing the \
       blocks.")
    .add_property("block_norm_threshold", 
      &bob::ip::HOG<double>::getBlockNormThreshold, 
      &bob::ip::HOG<double>::setBlockNormThreshold,
      "Threshold used to perform the clipping during the block normalization.")
    .def("resize", &bob::ip::HOG<double>::resize, 
      (arg("height"), arg("width")))
    .def("disable_block_normalization", 
      &bob::ip::HOG<double>::disableBlockNormalization)
    .def("get_output_shape", &bob::ip::HOG<double>::getOutputShape)
    .def("__call__", &hog_call1, (arg("input"), arg("output")),
      "Extract the HOG descriptors.")
    .def("__call__", &hog_call1_p, (arg("input")),
      "Extract the HOG descriptors.")
    .def("forward", &hog_call1, (arg("input"), arg("output")),
      "Extract the HOG descriptors.")
    .def("forward", &hog_call1_p, (arg("input")),
      "Extract the HOG descriptors.")
    .def("forward_", &hog_call2, (arg("input"), arg("output")),
      "Extract the HOG descriptors. This variant does not check the inputs.")
    .def("forward_", &hog_call2_p, (arg("input")),
      "Extract the HOG descriptors. This variant does not check the inputs.")
  ;
}
