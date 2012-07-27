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
#include <blitz/array.h>

#include "core/python/exception.h"

#include "core/array_assert.h"

#include "localization.h"
#include "detection.h"

namespace bp = boost::python;
namespace array = bob::core::array;

static void load(visioner::SWScanner& s, 
    const blitz::Array<int16_t,2>& grayimage) {
  array::assertZeroBase(grayimage);
  array::assertCContiguous(grayimage);
  bool err = s.load(grayimage.data(), grayimage.rows(), grayimage.columns());
  if (!err) PYTHON_ERROR(RuntimeError, "failed to load image at subwindow scanner");
}

static bp::tuple detect_max(visioner::Model& cmodel, size_t levels,
    visioner::SWScanner& cscanner) {

  // detect faces
  visioner::detection_t detection;
  visioner::detect_max(cmodel[0], levels, cscanner, detection);

  // Returns a tuple containing the detection bbox
  qreal x, y, width, height;
  detection.second.getRect(&x, &y, &width, &height);
  return bp::make_tuple(x, y, width, height, detection.first);
}

static bp::tuple detect(visioner::Model& cmodel, double threshold, size_t levels,
    float cluster, visioner::SWScanner& cscanner) {
  // detect faces
  visioner::detections_t detections;
  visioner::detect(cmodel[0], threshold, levels, cscanner, detections);

  // cluster detections
  visioner::nms(detections, cluster);

  // order detections by descending order
  visioner::sort_desc(detections);

  // Returns a tuple containing all detections, with descending scores
  bp::list tmp;
  qreal x, y, width, height;
  for (size_t i=0; i<detections.size(); ++i) {
    detections[i].second.getRect(&x, &y, &width, &height);
    tmp.append(bp::make_tuple(x, y, width, height, detections[i].first));
  }
  return bp::tuple(tmp);
}

static bp::tuple locate(visioner::Model& cmodel, visioner::Model& lmodel,
    size_t levels, visioner::SWScanner& cscanner, 
    visioner::SWScanner& lscanner) {
  // Locate keypoints
  visioner::points_t dt_points;
  visioner::rect_t dt_region;
  visioner::locate(cmodel, lmodel, levels, cscanner, lscanner, 
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

static bp::tuple load_model(const std::string& filename) {
  visioner::Model model;
  visioner::param_t param;
  if (visioner::load_model(param, model, filename) == false) {        
    PYTHON_ERROR(IOError, "failed to load the model");
  }
  return bp::make_tuple(model, param);
}

static void save_model(const visioner::Model& model,
    const visioner::param_t& param, const std::string& filename) {
  if (visioner::save_model(param, model, filename) == false) {        
    PYTHON_ERROR(IOError, "failed to save the model");
  }
}

void bind_visioner_localize() {
  //opaque, just needs to pass around.
  bp::class_<visioner::Model, boost::shared_ptr<visioner::Model> >("Model",
      "Multivariate model (a set of univariate models which are, in turn a sum of look-up tables)", bp::init<>("Default constructor (empty model)"));
  
  //this class holds the points found by the face localization
  bp::class_<visioner::param_t, boost::shared_ptr<visioner::param_t> >("param_t",
      "Parameters: (1) classification and localization models; (2) sliding-windows sampling; (3) features", bp::init<>("Default constructor"))
    .def_readwrite("ds", &visioner::param_t::m_ds)
    ;

  bp::class_<visioner::SWScanner, boost::shared_ptr<visioner::SWScanner> >("SWScanner", "Process the sub-windows of the given scaled images: (1) validated by the tagger or (2) processed by the model", bp::init<visioner::param_t>((bp::arg("parameters")), "Constructor"))
    .def("load", &load, (bp::arg("self"), bp::arg("image")), "Loads an int16 2D array into the scanner. You must convert the image to a 16-bit integer representation first.")
    ;

  bp::def("detect_max", &detect_max, (bp::arg("class_model"), bp::arg("levels"),
        bp::arg("class_scanner")), "Detects the most likely face on an image preloaded by the (classification) scanner. Returns a tuple with the detected region and associated score in the following order (x, y, width, height, score). All values are floating-point numbers.");

  bp::def("detect", &detect, (bp::arg("class_model"), bp::arg("threshold"), bp::arg("levels"),
        bp::arg("cluster"), bp::arg("class_scanner")), "Detects faces on an image preloaded by the (classification) scanner. Returns a tuple with the detected regions and associated scores in the following order (x, y, width, height, score). All values are floating-point numbers.");

  bp::def("locate", &locate, (bp::arg("class_model"), bp::arg("loc_model"), bp::arg("levels"),
        bp::arg("class_scanner"), bp::arg("loc_scanner")), "Locates faces on an image preloaded by the (classification and localization) scanners. Returns a tuple with the detected region and all detected landmarks");

  bp::def("load_model", &load_model, (bp::arg("filename")), "Loads the model and parameters from a given file.\n\n**Note**: Serialization will use a native text format by default. Files that have their names suffixed with '.gz' will be automatically decompressed. If the filename ends in '.vbin' or '.vbgz' the format used will be the native binary format.");
  bp::def("save_model", &save_model, (bp::arg("model"), bp::arg("parameters"), bp::arg("filename")), "Saves the model and parameters to a given file.\n\n**Note**: Serialization will use a native text format by default. Files that have their name suffixed with '.gz' will be automatically decompressed. If the filename ends in '.vbin' or '.vbgz' the format used will be the native binary format.");
}
