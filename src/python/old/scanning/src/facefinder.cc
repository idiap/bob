/**
 * @file python/old/scanning/src/facefinder.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds the FaceFinder to python
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
