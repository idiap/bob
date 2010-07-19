/**
 * @file src/scanning/scanner.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Scanner to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "ip/vision.h"
#include "ip/Image.h"
#include "scanning/Pattern.h"
#include "scanning/Explorer.h"
#include "scanning/explorers/MSExplorer.h"
#include "scanning/explorers/ContextExplorer.h"
#include "scanning/explorers/TrackContextExplorer.h"

using namespace boost::python;

void bind_scanning_explorer()
{
  class_<Torch::Explorer, bases<Torch::Object>, boost::noncopyable>("Explorer", "Explores the 4D (position, scale, confidence) space and decides which subwindows to scan at the next step", no_init)
    .def("clear", &Torch::Explorer::clear, (arg("self")), "Delete old detections")
    .def("init", (bool (Torch::Explorer::*)(int, int))&Torch::Explorer::init, (arg("self"), arg("image_width"), arg("image_height")), "Initializes the scanning process with the given image size")
    .def("init", (bool (Torch::Explorer::*)(const Torch::sRect2D&))&Torch::Explorer::init, (arg("self"), arg("roi")), "Initializes the scanning process with a specific region-of-interest")
    .def("preprocess", (bool (Torch::Explorer::*)(const Torch::Image&))&Torch::Explorer::preprocess, (arg("self"), arg("image")), "Preprocesses the image (extract features ...) => store data in <prune_ips> and <evaluation_ips>")
    .def("preprocess", (bool (Torch::Explorer::*)(void))&Torch::Explorer::preprocess, (arg("self")), "Preprocesses the image (checks for pattern's sub-windows)")
    .def("getNoScannedSWs", &Torch::Explorer::getNoScannedSWs, (arg("self")), "Returns the number of scanned subwindows")
    .def("getNoPrunnedSWs", &Torch::Explorer::getNoScannedSWs, (arg("self")), "Returns the number of subwindows after prunning")
    .def("getNoAcceptedSWs", &Torch::Explorer::getNoScannedSWs, (arg("self")), "Returns the final number of accepted subwindows")
    .def("getPatterns", &Torch::Explorer::getPatterns, return_internal_reference<>(), (arg("self")), "Returns the final number of patterns")
    .def("getModelWidth", &Torch::Explorer::getModelWidth, (arg("self")), "Returns the current model width")
    .def("getModelHeight", &Torch::Explorer::getModelHeight, (arg("self")), "Returns the current model height")
    .def("getNoScales", &Torch::Explorer::getNoScales, (arg("self")), "Returns the number of search scales")
    .def("getScale", &Torch::Explorer::getScale, return_internal_reference<>(), (arg("self"), arg("index")), "Returns a specific scale")
    ;

  class_<Torch::MSExplorer, bases<Torch::Explorer> >("MSExplorer", "Multiscale explorer, keeps the image size and varies the size of the model/template (window scan)", init<>())
    ;

  enum_<Torch::ContextExplorer::Mode>("Mode")
    .value("Scanning", Torch::ContextExplorer::Scanning)
    .value("Profiling", Torch::ContextExplorer::Profiling)
    ;

  class_<Torch::ContextExplorer, bases<Torch::MSExplorer> >("ContextExplorer", "searches the 4D scanning space using a greedy method based on	a context-based model to remove false alarms and drives iteratively the detections to better locations", init<Torch::ContextExplorer::Mode>(arg("mode")=Torch::ContextExplorer::Scanning, "Initializes the context explorer defining the operational mode"))
    .def("getMode", &Torch::ContextExplorer::getMode, arg("self"), "Returns the current operational mode")
    .def("setMode", &Torch::ContextExplorer::setMode, (arg("self"), arg("mode")), "Sets the current operational mode")
    ;

  class_<Torch::TrackContextExplorer, bases<Torch::ContextExplorer> >("TrackContextExplorer", "A ContextExplorer that processes only a specified target subwindow", init<Torch::ContextExplorer::Mode>(arg("mode")=Torch::ContextExplorer::Scanning, "Initializes the context explorer defining the operational mode"))
    .def("setSeedPatterns", &Torch::TrackContextExplorer::setSeedPatterns, (arg("self"), arg("pattern_list")), "Changes the subwindows to process.")
    ;
}
