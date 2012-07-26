/**
 * @file python/visioner/src/localize.cc
 * @date Thu Jul 21 13:13:06 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Face localization bridge for Visioner
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

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "core/python/ndarray.h"

#include "visioner/util/util.h"
#include "visioner/cv/cv_detector.h"
#include "visioner/cv/cv_localizer.h"

namespace bp = boost::python;
namespace tp = bob::python;

static bp::tuple detect_max(bob::visioner::CVDetector& det, 
    tp::const_ndarray image) {

  blitz::Array<bob::visioner::grey_t,2> bzimage = image.bz<bob::visioner::grey_t,2>();
  det.load(bzimage.data(), bzimage.rows(), bzimage.cols());
  bob::visioner::detections_t detections;
  det.scan(detections);
  det.sort_desc(detections);

  // Returns a tuple containing the detection bbox
  qreal x, y, width, height;
  detections[0].second.first.getRect(&x, &y, &width, &height);
  return bp::make_tuple(x, y, width, height, detections[0].first);
}

static bp::tuple detect(bob::visioner::CVDetector& det,
    tp::const_ndarray image) {
  
  blitz::Array<bob::visioner::grey_t,2> bzimage = image.bz<bob::visioner::grey_t,2>();
  det.load(bzimage.data(), bzimage.rows(), bzimage.cols());
  bob::visioner::detections_t detections;
  det.scan(detections);
  det.sort_desc(detections);

  // Returns a tuple containing all detections, with descending scores
  bp::list tmp;
  qreal x, y, width, height;
  for (size_t i=0; i<detections.size(); ++i) {
    detections[i].second.first.getRect(&x, &y, &width, &height);
    tmp.append(bp::make_tuple(x, y, width, height, detections[i].first));
  }
  return bp::tuple(tmp);
}

/**
static bp::tuple locate(bob::visioner::Model& cmodel, bob::visioner::Model& lmodel,
    size_t levels, bob::visioner::SWScanner& cscanner, 
    bob::visioner::SWScanner& lscanner) {
  // Locate keypoints
  bob::visioner::points_t dt_points;
  bob::visioner::rect_t dt_region;
  bob::visioner::locate(cmodel, lmodel, levels, cscanner, lscanner, 
      dt_points, dt_region);

  // Returns a 2-tuple: 
  // [0] => The region bounding box as x, y, width, height
  // [1] => A tuple containing all points detected
  qreal x, y, width, height;
  dt_region.getRect(&x, &y, &width, &height);
  bp::tuple bbox = bp::make_tuple(x, y, width, height);
  
  bp::list tmp;
  for (size_t i=0; i<dt_points.size(); ++i) {
    tmp.append(bp::make_tuple(dt_points[i].x(), dt_points[i].y()));
  }

  return bp::make_tuple(bbox, bp::tuple(tmp));
}
**/

void bind_visioner_localize() {
  bp::enum_<bob::visioner::CVDetector::Type>("DetectionMethod")
    .value("Scanning", bob::visioner::CVDetector::Scanning)
    .value("GroundTruth", bob::visioner::CVDetector::GroundTruth)
    ;

  bp::class_<bob::visioner::CVDetector>("CVDetector", "Object detector that processes a pyramid of images", bp::init<const std::string&, bob::visioner::scalar_t, bob::visioner::index_t, bob::visioner::index_t, bob::visioner::scalar_t, bob::visioner::CVDetector::Type>((bp::arg("model"), bp::arg("threshold")=0.0, bp::arg("levels")=0, bp::arg("scale_variation")=2, bp::arg("clustering")=0.05, bp::arg("detection_method")=bob::visioner::CVDetector::GroundTruth), "Basic constructor with the following parameters:\n\nmodel\n  file containing the model to be loaded; **note**: Serialization will use a native text format by default. Files that have their names suffixed with '.gz' will be automatically decompressed. If the filename ends in '.vbin' or '.vbgz' the format used will be the native binary format.\n\nthreshold\n  object classification threshold\n\nlevels\n  levels (the more, the faster)\n\nscale_variation\n  scale variation in pixels\n\nclustering\n  overlapping threshold for clustering detections\n\ndetection_method\n  Scanning or GroundTruth"))
    .def_readwrite("threshold", &bob::visioner::CVDetector::m_threshold, "Object classification threshold")
    .def_readwrite("levels", &bob::visioner::CVDetector::m_levels, "Levels (the more, the faster)")
    .def_readwrite("scale_variation", &bob::visioner::CVDetector::m_ds, "Scale variation in pixels")
    .def_readwrite("clustering", &bob::visioner::CVDetector::m_cluster, "Overlapping threshold for clustering detections")
    .def_readwrite("type", &bob::visioner::CVDetector::m_type, "Scanning or GroundTruth")
    .def("detect", &detect, (bp::arg("self"), bp::arg("image")), "Detects faces in the input (gray-scaled) image according to the current settings")
    .def("detect_max", &detect, (bp::arg("self"), bp::arg("image")), "Detects the most probable face in the input (gray-scaled) image according to the current settings")
    .def("save", &bob::visioner::CVDetector::save, (bp::arg("self"), bp::arg("filename")), "Saves the model and parameters to a given file.\n\n**Note**: Serialization will use a native text format by default. Files that have their name suffixed with '.gz' will be automatically decompressed. If the filename ends in '.vbin' or '.vbgz' the format used will be the native binary format.")
    ;

  /**
  bp::def("locate", &locate, (bp::arg("class_model"), bp::arg("loc_model"), bp::arg("levels"),
        bp::arg("class_scanner"), bp::arg("loc_scanner")), "Locates faces on an image preloaded by the (classification and localization) scanners. Returns a tuple with the detected region and all detected landmarks");
  **/
}
