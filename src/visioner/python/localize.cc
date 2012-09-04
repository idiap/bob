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

#include "bob/core/python/ndarray.h"

#include "bob/visioner/util/util.h"
#include "bob/visioner/cv/cv_detector.h"
#include "bob/visioner/cv/cv_localizer.h"

namespace bp = boost::python;
namespace tp = bob::python;

static bp::object detect_max(bob::visioner::CVDetector& det, 
    tp::const_ndarray image) {

  blitz::Array<uint8_t,2> bzimage = image.bz<uint8_t,2>();
  det.load(bzimage.data(), bzimage.rows(), bzimage.cols());
  std::vector<bob::visioner::detection_t> detections;
  det.scan(detections);

  if (detections.size() == 0) {
    return bp::object();
  }

  det.sort_desc(detections);

  // Returns a tuple containing the detection bbox
  qreal x, y, width, height;
  detections[0].second.first.getRect(&x, &y, &width, &height);
  return bp::make_tuple(x, y, width, height, detections[0].first);
}

static bp::object detect(bob::visioner::CVDetector& det,
    tp::const_ndarray image) {
  
  blitz::Array<uint8_t,2> bzimage = image.bz<uint8_t,2>();
  det.load(bzimage.data(), bzimage.rows(), bzimage.cols());
  std::vector<bob::visioner::detection_t> detections;
  det.scan(detections);
  
  if (detections.size() == 0) {
    return bp::object();
  }

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

static bp::object locate(bob::visioner::CVLocalizer& loc,
    bob::visioner::CVDetector& det, tp::const_ndarray image) {

  blitz::Array<uint8_t,2> bzimage = image.bz<uint8_t,2>();
  det.load(bzimage.data(), bzimage.rows(), bzimage.cols());
  std::vector<bob::visioner::detection_t> detections;
  det.scan(detections);
  
  if (detections.size() == 0) {
    return bp::object();
  }

  det.sort_desc(detections);

  // Locate keypoints
  bob::visioner::Object object;
  std::vector<QPointF> dt_points;

  for (std::vector<bob::visioner::detection_t>::const_iterator it = detections.begin(); it != detections.end(); ++ it) {
    if (det.match(*it, object) && loc.locate(det, it->second.first, dt_points))
      break;
  }

  // Returns a 2-tuple: 
  // [0] => The region bounding box as x, y, width, height
  // [1] => A tuple containing all points detected
  qreal x, y, width, height;
  detections[0].second.first.getRect(&x, &y, &width, &height);
  bp::tuple bbox = bp::make_tuple(x, y, width, height);
  
  bp::list tmp;
  for (size_t i=0; i<dt_points.size(); ++i) {
    tmp.append(bp::make_tuple(dt_points[i].x(), dt_points[i].y()));
  }

  return bp::make_tuple(bbox, bp::tuple(tmp));
}

void bind_visioner_localize() {
  bp::enum_<bob::visioner::CVDetector::Type>("DetectionMethod")
    .value("Scanning", bob::visioner::CVDetector::Scanning)
    .value("GroundTruth", bob::visioner::CVDetector::GroundTruth)
    ;

  bp::class_<bob::visioner::CVDetector>("CVDetector", "Object detector that processes a pyramid of images", bp::init<const std::string&, double, uint64_t, uint64_t, double, bob::visioner::CVDetector::Type>((bp::arg("model"), bp::arg("threshold")=0.0, bp::arg("scanning_levels")=0, bp::arg("scale_variation")=2, bp::arg("clustering")=0.05, bp::arg("method")=bob::visioner::CVDetector::GroundTruth), "Basic constructor with the following parameters:\n\nmodel\n  file containing the model to be loaded; **note**: Serialization will use a native text format by default. Files that have their names suffixed with '.gz' will be automatically decompressed. If the filename ends in '.vbin' or '.vbgz' the format used will be the native binary format.\n\nthreshold\n  object classification threshold\n\nscanning_levels\n  scanning levels (the more, the faster)\n\nscale_variation\n  scale variation in pixels\n\nclustering\n  overlapping threshold for clustering detections\n\nmethod\n  Scanning or GroundTruth"))
    .def_readwrite("threshold", &bob::visioner::CVDetector::m_threshold, "Object classification threshold")
    .add_property("scanning_levels", &bob::visioner::CVDetector::get_scan_levels, &bob::visioner::CVDetector::set_scan_levels, "Levels (the more, the faster)")
    .def_readwrite("scale_variation", &bob::visioner::CVDetector::m_ds, "Scale variation in pixels")
    .def_readwrite("clustering", &bob::visioner::CVDetector::m_cluster, "Overlapping threshold for clustering detections")
    .def_readwrite("method", &bob::visioner::CVDetector::m_type, "Scanning or GroundTruth (default)")
    .def("detect", &detect, (bp::arg("self"), bp::arg("image")), "Detects faces in the input (gray-scaled) image according to the current settings. The input image format should be a 2D array of dtype=uint8.")
    .def("detect_max", &detect_max, (bp::arg("self"), bp::arg("image")), "Detects the most probable face in the input (gray-scaled) image according to the current settings")
    .def("save", &bob::visioner::CVDetector::save, (bp::arg("self"), bp::arg("filename")), "Saves the model and parameters to a given file.\n\n**Note**: Serialization will use a native text format by default. Files that have their name suffixed with '.gz' will be automatically decompressed. If the filename ends in '.vbin' or '.vbgz' the format used will be the native binary format.")
    ;

  bp::enum_<bob::visioner::CVLocalizer::Type>("LocalizationMethod")
    .value("SingleShot", bob::visioner::CVLocalizer::SingleShot)
    .value("MultipleShots_Average", bob::visioner::CVLocalizer::MultipleShots_Average)
    .value("MultipleShots_Median", bob::visioner::CVLocalizer::MultipleShots_Median)
    ;

  bp::class_<bob::visioner::CVLocalizer>("CVLocalizer", "Keypoint localizer to be applied in tandem with ground-truth or detections from CVDetector", bp::init<const std::string&, bob::visioner::CVLocalizer::Type>((bp::arg("model"), bp::arg("method")=bob::visioner::CVLocalizer::MultipleShots_Median), "Basic constructor taking a model file and the localization method to use"))
      .def_readwrite("method", &bob::visioner::CVLocalizer::m_type, "SingleShot, MultipleShots_Average or MultipleShots_Median (default)")
      .def("locate", &locate, (bp::arg("self"), bp::arg("detector"), bp::arg("image")), "Runs the keypoint localization on the first (highest scored) face location determined by the detector. The input image format should be a 2D array of dtype=uint8.")
    .def("save", &bob::visioner::CVLocalizer::save, (bp::arg("self"), bp::arg("filename")), "Saves the model and parameters to a given file.\n\n**Note**: Serialization will use a native text format by default. Files that have their name suffixed with '.gz' will be automatically decompressed. If the filename ends in '.vbin' or '.vbgz' the format used will be the native binary format.")
    ;
}
