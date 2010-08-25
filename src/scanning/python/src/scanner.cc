/**
 * @file src/scanning/scanner.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Scanner to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "ip/Image.h"
#include "scanning/Scanner.h"
#include "scanning/TrackContextExplorer.h"

using namespace boost::python;

/**
 * A helper function to aid you in setting ROIs.
 *
 * @param scanner The scanner object where to set the ROIs
 * @param ref_image The reference image so we don't go over the borders
 * @param pattern Use this pattern as basis for the ROI location and size
 * @param increase_ratio This defines how much bigger the ROI will be compared
 * to the pattern
 *
 * @return A boolean indicating success (true) or failure (false)
 */
static bool set_roi(Torch::Scanner& scanner, Torch::Image& ref_image, 
    const Torch::Pattern& pattern, float increase_ratio)
{
  int width_increase = pattern.m_w * increase_ratio;
  int height_increase = pattern.m_h * increase_ratio;
  int width = pattern.m_w + width_increase;
  int height = pattern.m_w + height_increase;
  int x = pattern.m_x - width_increase/2;
  int y = pattern.m_y - height_increase/2;

  //some minor adjustments in case our ROI goes beyond image boundaries
  if (x < 0) {
    width += x; //removes 'x' from the window (NB: x is negative)
    x = 0;
  }
  if (y < 0) {
    height += y; //removes 'y' from the window (NB: y is negative)
    y = 0;
  }
  int im_width = ref_image.getWidth();
  int im_height = ref_image.getHeight();
  if ((x + width) > im_width) {
    width -= ((x + width) - im_width); //removes excess
  }
  if ((y + height) > im_height) {
    height -= ((y + height) - im_height); //removes excess
  }
  return scanner.addROI(x, y, width, height);
}

static bool set_rois(Torch::Scanner& scanner, Torch::Image& ref_image,
    Torch::PatternList& plist, float increase_ratio)
{
  bool success = true;
  for(int i=0; i<plist.size(); ++i) {
    success &= set_roi(scanner, ref_image, plist.get(i), increase_ratio);
  }
  return success;
}

static bool set_rois_from_last(Torch::Scanner& scanner, Torch::Image& ref_image,
    float increase_ratio)
{
  bool success = true;
  const Torch::PatternList& plist = scanner.getPatterns();
  for(int i=0; i<plist.size(); ++i) {
    success &= set_roi(scanner, ref_image, plist.get(i), increase_ratio);
  }
  return success;
}

static Torch::TrackContextExplorer* scanner_get_trackctx_explorer(Torch::Scanner& scanner)
{
  return dynamic_cast<Torch::TrackContextExplorer*>(&scanner.getExplorer());
}

void bind_scanning_scanner()
{
  class_<Torch::Scanner, bases<Torch::Object>, boost::noncopyable>("Scanner", "Objects of this type scan an image for rectangular patterns in 4D (position + scale + model confidence) scanning space. They use a Explorer object to investigate the 4D pattern space and a Selector object to select the best patterns from the candidates.", no_init)
    .def("addROI", (bool (Torch::Scanner::*)(const Torch::sRect2D&))&Torch::Scanner::addROI, (arg("self"), arg("window")), "Adds another Region-of-Interest to shorten the search area")
    .def("addROI", (bool (Torch::Scanner::*)(int, int, int, int))&Torch::Scanner::addROI, (arg("self"), arg("x"), arg("y"), arg("width"), arg("height")), "Adds another Region-of-Interest to shorten the search area (coordinates and sizes)")
    .def("addROI", &set_roi, (arg("self"), arg("reference_image"), arg("pattern"), arg("increase_ratio")), "Adds a new RoI based on a pattern and a reference image. The new RoI will be increased according to the ratio specified (independently in both 'x' and 'y' directions. The control image is used to limite the RoI expansion out of the image boundaries.") 
    .def("addROIs", &set_rois, (arg("self"), arg("reference_image"), arg("pattern_list"), arg("increase_ratio")), "Adds new RoIs based on a pattern list and a reference image. The new RoIs will be increased according to the ratio specified (independently in both 'x' and 'y' directions. The control image is used to limite the RoI expansion out of the image boundaries.") 
    .def("addROIs", &set_rois_from_last, (arg("self"), arg("reference_image"), arg("increase_ratio")), "Adds new RoIs based on the previously calculated pattern list and a reference image. The new RoIs will be increased according to the ratio specified (independently in both 'x' and 'y' directions. The control image is used to limite the RoI expansion out of the image boundaries.")
    .def("deleteROI", &Torch::Scanner::deleteROI, (arg("self"), arg("index")), "Deletes a specific RoI from my internal list")
    .def("deleteAllROIs", &Torch::Scanner::deleteAllROIs, (arg("self")), "Removes all RoIs from my internal list")
    .def("init", &Torch::Scanner::init, (arg("self"), arg("image")), "Initializes the scanner")
    .def("process", &Torch::Scanner::process, (arg("self"), arg("image")), "Processes a single image")
    .def("getNoROIs", &Torch::Scanner::getNoROIs, (arg("self")), "Returns the number of RoIs currently set")
    .def("getROI", &Torch::Scanner::getROI, return_internal_reference<>(), (arg("self"), arg("index")), "Returns a specific RoI previously set")
    .def("getNoScannedSWs", &Torch::Scanner::getNoScannedSWs, arg("self"), "Returns the number of scanned selection-windows")
    .def("getNoPrunnedSWs", &Torch::Scanner::getNoPrunnedSWs, arg("self"), "Returns the number of prunned selection-windows")
    .def("getNoAcceptedSWs", &Torch::Scanner::getNoAcceptedSWs, arg("self"), "Returns the number of accepted selection-windows")
    .def("getPatterns", &Torch::Scanner::getPatterns, return_internal_reference<>(), arg("self"), "Returns all patterns found on the previous scan")
    .def("getExplorer", (Torch::Explorer& (Torch::Scanner::*)(void))&Torch::Scanner::getExplorer, return_internal_reference<>(), arg("self"), "Returns the current explorer configured")
    .def("tryGetTrackContextExplorer", &scanner_get_trackctx_explorer, return_internal_reference<>(), arg("self"), "This method will return the currently configured TrackContextExplorer if that is the case, otherwise None")
    ;
}
