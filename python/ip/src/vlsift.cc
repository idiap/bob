/**
 * @file python/ip/src/vlsift.cc
 * @date Mon Dec 19 17:19:07 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds VLFEAT SIFT features to python
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
#include "ip/VLSIFT.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ip = bob::ip;
namespace ca = bob::core::array;


static object call_vlsift(ip::VLSIFT& op, tp::const_ndarray src) {
  const ca::typeinfo& info = src.type();  
  if (info.nd != 2) PYTHON_ERROR(TypeError, "sift features extractor does not support input of type '%s'", info.str().c_str());
  if(info.dtype != ca::t_uint8)
    PYTHON_ERROR(TypeError, "sift features does not support type '%s'", info.str().c_str());

  std::vector<blitz::Array<double,1> > dst;
  op(src.bz<uint8_t,2>(), dst);
  list t;
  for(size_t i=0; i<dst.size(); ++i) t.append(dst[i]);
  return t;
}


void bind_ip_vlsift() {
  static const char* VLSIFT_doc = "Computes SIFT features using the VLFeat library";

  class_<ip::VLSIFT, boost::shared_ptr<ip::VLSIFT> >("VLSIFT", VLSIFT_doc, init<const int, const int, const int, const int, const int, optional<const double, const double, const double> >((arg("height"), arg("width"), arg("n_intervals"), arg("n_octaves"), arg("octave_min"), arg("peak_thres")=10, arg("edge_thres")=0.03, arg("magnif")=3), "Creates an object to compute SIFT features"))
    .def("__call__", &call_vlsift, (arg("self"), arg("src")), "Computes the SIFT features from an input image. It returns a list of descriptors, one for each keypoint and orientation. The first four values are the x, y, sigma and orientation of the values. The 128 remaining values define the descriptor.")
    ;
}
