/**
 * @file src/core/file.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the File object type into python 
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "core/Tensor.h"
#include "core/File.h"

using namespace boost::python;

void bind_core_file()
{
  class_<Torch::File, bases<Torch::Object> >("File", init<>())
    .def("open", (bool (Torch::File::*)(const char*, const char*))&Torch::File::open)
    .def("open", (bool (Torch::File::*)(FILE*))&Torch::File::open)
    .def("close", &Torch::File::close) 
    .def("isOpened", &Torch::File::isOpened) 
    .def("eof", &Torch::File::eof)
    .def("flush", &Torch::File::flush)
    .def("seek", &Torch::File::seek)
    .def("tell", &Torch::File::tell)
    ;
}
