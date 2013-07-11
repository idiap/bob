/**
 * @file ip/python/vlsift.cc
 * @date Mon Dec 19 17:19:07 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds VLFEAT SIFT features to python
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
#include "bob/ip/VLSIFT.h"

using namespace boost::python;

static object call_vlsift(bob::ip::VLSIFT& op, bob::python::const_ndarray src) 
{
  std::vector<blitz::Array<double,1> > dst;
  op(src.bz<uint8_t,2>(), dst);
  list t;
  for(size_t i=0; i<dst.size(); ++i) t.append(dst[i]);
  return t;
}

static object call_kp_vlsift(bob::ip::VLSIFT& op, bob::python::const_ndarray src, bob::python::const_ndarray kp) 
{
  std::vector<blitz::Array<double,1> > dst;
  op(src.bz<uint8_t,2>(), kp.bz<double,2>(), dst);
  list t;
  for(size_t i=0; i<dst.size(); ++i) t.append(dst[i]);
  return t;
}

void bind_ip_vlsift() 
{
  static const char* VLSIFT_doc = "Computes SIFT features using the VLFeat library";

  class_<bob::ip::VLSIFT, boost::shared_ptr<bob::ip::VLSIFT> >("VLSIFT", VLSIFT_doc, init<const size_t, const size_t, const size_t, const size_t, const int, optional<const double, const double, const double> >((arg("self"), arg("height"), arg("width"), arg("n_intervals"), arg("n_octaves"), arg("octave_min"), arg("peak_thres")=0.03, arg("edge_thres")=10., arg("magnif")=3.), "Creates an object to compute SIFT features using the VLFeat library"))
    .def(init<bob::ip::VLSIFT&>((arg("self"), arg("other"))))
    .def(self == self)
    .def(self != self)
    .add_property("height", &bob::ip::VLSIFT::getHeight, &bob::ip::VLSIFT::setHeight, "The height of the image to process")
    .add_property("width", &bob::ip::VLSIFT::getWidth, &bob::ip::VLSIFT::setWidth, "The width of the image to process")
    .add_property("n_intervals", &bob::ip::VLSIFT::getNIntervals, &bob::ip::VLSIFT::setNIntervals, "The number of intervals in each octave")
    .add_property("n_octaves", &bob::ip::VLSIFT::getNOctaves, &bob::ip::VLSIFT::setNOctaves, "The number of intervals in each octave")
    .add_property("octave_min", &bob::ip::VLSIFT::getOctaveMin, &bob::ip::VLSIFT::setOctaveMin, "The index of the minimum octave")
    .add_property("peak_thres", &bob::ip::VLSIFT::getPeakThres, &bob::ip::VLSIFT::setPeakThres, "The peak threshold (minimum amount of contrast to accept a keypoint)")
    .add_property("edge_thres", &bob::ip::VLSIFT::getEdgeThres, &bob::ip::VLSIFT::setEdgeThres, "The edge rejection threshold")
    .add_property("magnif", &bob::ip::VLSIFT::getMagnif, &bob::ip::VLSIFT::setMagnif, "The magnification factor (descriptor size is determined by multiplying the keypoint scale by this factor)")
    .def("__call__", &call_vlsift, (arg("self"), arg("src")), "Computes the SIFT features from an input image (by first detecting keypoints). It returns a list of descriptors, one for each keypoint and orientation. The first four values are the x, y, sigma and orientation of the values. The 128 remaining values define the descriptor.")
    .def("__call__", &call_kp_vlsift, (arg("self"), arg("src"), arg("keypoints")), "Computes the SIFT features from an input image and a set of keypoints. A keypoint is specified by a 3- or 4-tuple (y, x, sigma, [orientation]). The orientation is estimated if not specified. It returns a list of descriptors, one for each keypoint and orientation. The first four values are the x, y, sigma and orientation of the values. The 128 remaining values define the descriptor.")
    ;
}
