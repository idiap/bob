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
  class_<Torch::spCore, bases<Torch::Object>, boost::noncopyable>("spCore", "A base type for all signal processing operators", no_init)
    .def("loadFile", &Torch::spCore::loadFile, (arg("self"), arg("file")), "Loads the content from files (not options)")
    .def("saveFile", &Torch::spCore::saveFile, (arg("self"), arg("file")), "Saves content into file (not options)")
    .def("process", &Torch::spCore::process, (arg("self"), arg("input")), "Processes an input tensor, stores the results internally")
    .def("setRegion", &Torch::spCore::setRegion, (arg("self"), arg("region")), "Subselects a region to be processed instead of the whole input tensor")
    .def("setModelSize", &Torch::spCore::setModelSize, (arg("self"), arg("model_size")), "changes the model size (if used with some machine)")
    .def("getID", &Torch::spCore::getID, (arg("self")), "Gets my own identifier")
    .def("getNOutputs", &Torch::spCore::getNOutputs, (arg("self")), "Returns the number of outputs generated during the processing")
    .def("getOutput", &Torch::spCore::getOutput, return_internal_reference<>(), (arg("self"), arg("index")), "Gets a specific output tensor.")
    ;
}
