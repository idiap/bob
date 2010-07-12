/**
 * @file src/scanning/facefinder.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the FaceFinder to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "ip/Image.h"
#include "scanning/FaceFinder.h"

using namespace boost::python;

void bind_scanning_facefinder()
{
  class_<Torch::FaceFinder, bases<Torch::Object> >("FaceFinder", 
      init<optional<const char* > >())
    .def("reset", (bool (Torch::FaceFinder::*)(const char*))&Torch::FaceFinder::reset)
    .def("process", &Torch::FaceFinder::process)
    .def("getPatterns", &Torch::FaceFinder::getPatterns, return_internal_reference<>())
    ;
}
