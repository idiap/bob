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
  for(unsigned int i=0; i<plist.size(); ++i) {
    success &= set_roi(scanner, ref_image, plist.get(i), increase_ratio);
  }
  return success;
}

static bool set_rois_from_last(Torch::Scanner& scanner, Torch::Image& ref_image,
    float increase_ratio)
{
  bool success = true;
  const Torch::PatternList& plist = scanner.getPatterns();
  for(unsigned int i=0; i<plist.size(); ++i) {
    success &= set_roi(scanner, ref_image, plist.get(i), increase_ratio);
  }
  return success;
}

void bind_scanning_scanner()
{
  class_<Torch::Scanner, bases<Torch::Object>, boost::noncopyable>("Scanner", no_init)
    .def("addROI", (bool (Torch::Scanner::*)(const Torch::sRect2D&))&Torch::Scanner::addROI)
    .def("addROI", (bool (Torch::Scanner::*)(int, int, int, int))&Torch::Scanner::addROI)
    .def("addROI", &set_roi) 
    .def("addROIs", &set_rois) 
    .def("addROIs", &set_rois_from_last)
    .def("deleteROI", &Torch::Scanner::deleteROI)
    .def("deleteAllROIs", &Torch::Scanner::deleteAllROIs)
    .def("init", &Torch::Scanner::init)
    .def("process", &Torch::Scanner::process)
    .def("getNoROIs", &Torch::Scanner::getNoROIs)
    .def("getROI", &Torch::Scanner::getROI, return_internal_reference<>())
    .def("getNoScannedSWs", &Torch::Scanner::getNoScannedSWs)
    .def("getNoPrunnedSWs", &Torch::Scanner::getNoPrunnedSWs)
    .def("getNoAcceptedSWs", &Torch::Scanner::getNoAcceptedSWs)
    .def("getPatterns", &Torch::Scanner::getPatterns, return_internal_reference<>())
    ;
}
