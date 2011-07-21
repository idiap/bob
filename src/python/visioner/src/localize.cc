/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 21 Jul 2011 12:32:02 CEST
 *
 * @brief Face localization bridge for Visioner
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <blitz/array.h>

#include "core/array_assert.h"
#include "core/python/exception.h"

#include "visioner/proc/localization.h"

using namespace boost::python;
namespace array = Torch::core::array;

static void load(visioner::SWScanner& s, 
    const blitz::Array<int16_t,2>& grayimage) {
  array::assertZeroBase(grayimage);
  array::assertCContiguous(grayimage);
  bool err = s.load(grayimage.data(), grayimage.rows(), grayimage.columns());
  if (!err) PYTHON_ERROR(RuntimeError, "failed to load image at subwindow scanner");
}

static tuple locate(visioner::Model& cmodel, visioner::Model& lmodel,
    size_t levels, visioner::SWScanner& cscanner, 
    visioner::SWScanner& lscanner) {
  const visioner::fimage_info_t& iinfo = lscanner.iinfos()[0];
  if (iinfo.m_objects.empty()) return tuple();

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
  tuple bbox = make_tuple(x, y, width, height);
  
  list tmp;
  for (size_t i=0; i<dt_points.size(); ++i) {
    tmp.append(make_tuple(dt_points[i].x(), dt_points[i]));
  }

  return make_tuple(bbox, tmp);
}

void bind_visioner_localize() {
  //opaque, just needs to pass around.
  class_<visioner::Model, boost::shared_ptr<visioner::Model> >("Model",
      "Multivariate model (a set of univariate models which are, in turn a sum of look-up tables)", init<>("Default constructor (empty model)"));
  
  //this class holds the points found by the face localization
  class_<visioner::param_t, boost::shared_ptr<visioner::param_t> >("param_t",
      "Parameters: (1) classification and localization models; (2) sliding-windows sampling; (3) features", init<>("Default constructor"))
    .def_readwrite("ds", &visioner::param_t::m_ds)
    ;

  class_<visioner::SWScanner, boost::shared_ptr<visioner::SWScanner> >("SWScanner", "Process the sub-windows of the given scaled images: (1) validated by the tagger or (2) processed by the model", init<visioner::param_t>((arg("parameters")), "Constructor"))
    .def("load", &load, (arg("image")), "Loads an int16 2D array into the scanner. You must convert the image to a 16-bit integer representation first.")
    ;

  def("locate", &locate, (arg("cmodel"), arg("lmodel"), arg("levels"),
        arg("cscanner"), arg("lscanner")), "Locates faces on an image preloaded by the scanners. Returns a tuple with the detected region and all detected landmarks");
}
