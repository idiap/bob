/**
 * @file src/scanning/facefinder.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the FaceFinder to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "ip/Image.h"
#include "scanning/Scanner.h"
#include "scanning/FaceFinder.h"

using namespace boost::python;

void bind_scanning_facefinder()
{
  class_<Torch::FaceFinder, bases<Torch::Object> >("FaceFinder", "The FaceFinder locates faces in images. This type groups together the main scanning objects to make the face detection system easier to use.", init<optional<const char* > >((arg("filename")), "Builds a new face finder out of a parameter file."))
    .def("reset", (bool (Torch::FaceFinder::*)(const char*))&Torch::FaceFinder::reset, (arg("self"), arg("filename")), "Change the scanning parameters")
    .def("process", &Torch::FaceFinder::process, (arg("self"), arg("image")), "Scans for patterns in the given image")
    .def("getPatterns", &Torch::FaceFinder::getPatterns, return_internal_reference<>(), (arg("self")), "Returns patterns from the last scanned image")
    .def("getScanner", (Torch::Scanner& (Torch::FaceFinder::*)())&Torch::FaceFinder::getScanner, return_internal_reference<>(), (arg("self")), "Returns the current scanner being used")
    ;
}
