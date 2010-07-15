/**
 * @file src/core/spcore.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the spCore object type into python 
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "core/Tensor.h"
#include "core/File.h"
#include "core/spCore.h"

using namespace boost::python;

void bind_core_spcore()
{
  class_<Torch::spCore, bases<Torch::Object>, boost::noncopyable>("spCore", no_init)
    .def("loadFile", &Torch::spCore::loadFile)
    .def("saveFile", &Torch::spCore::saveFile)
    .def("process", &Torch::spCore::process)
    .def("setRegion", &Torch::spCore::setRegion)
    .def("setModelSize", &Torch::spCore::setModelSize)
    .def("getID", &Torch::spCore::getID)
    .def("getNOutputs", &Torch::spCore::getNOutputs)
    .def("getOutput", &Torch::spCore::getOutput, return_internal_reference<>())
    ;
}
