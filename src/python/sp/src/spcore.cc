/**
 * @file python/sp/src/spcore.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds the spCore object type into python
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "core/Object.h"
#include "core/Tensor.h"
#include "core/File.h"
#include "sp/spCore.h"

using namespace boost::python;

void bind_sp_spcore()
{
  class_<bob::spCore, bases<bob::Object>, boost::noncopyable>("spCore", "A base type for all signal processing operators", no_init)
    .def("loadFile", &bob::spCore::loadFile, (arg("self"), arg("file")), "Loads the content from files (not options)")
    .def("saveFile", &bob::spCore::saveFile, (arg("self"), arg("file")), "Saves content into file (not options)")
    .def("process", &bob::spCore::process, (arg("self"), arg("input")), "Processes an input tensor, stores the results internally")
    .def("setRegion", &bob::spCore::setRegion, (arg("self"), arg("region")), "Subselects a region to be processed instead of the whole input tensor")
    .def("setModelSize", &bob::spCore::setModelSize, (arg("self"), arg("model_size")), "changes the model size (if used with some machine)")
    .def("getID", &bob::spCore::getID, (arg("self")), "Gets my own identifier")
    .def("getNOutputs", &bob::spCore::getNOutputs, (arg("self")), "Returns the number of outputs generated during the processing")
    .def("getOutput", &bob::spCore::getOutput, return_internal_reference<>(), (arg("self"), arg("index")), "Gets a specific output tensor.")
    ;
}
