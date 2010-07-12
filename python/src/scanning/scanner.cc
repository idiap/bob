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

void bind_scanning_scanner()
{
  class_<Torch::Scanner, bases<Torch::Object>, boost::noncopyable>("Scanner", no_init)
    .def("addROI", (bool (Torch::Scanner::*)(const Torch::sRect2D&))&Torch::Scanner::addROI)
    .def("addROI", (bool (Torch::Scanner::*)(int, int, int, int))&Torch::Scanner::addROI)
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
