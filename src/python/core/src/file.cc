/**
 * @file src/python/core/src/file.cc
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
  class_<Torch::File, bases<Torch::Object> >("File", init<>("Base class for file reading and writing in Torch"))
    .def("open", (bool (Torch::File::*)(const char*, const char*))&Torch::File::open, (arg("self"), arg("filename"), arg("mode")), "Opens a file to read or write")
    .def("open", (bool (Torch::File::*)(FILE*))&Torch::File::open, (arg("self"), arg("opened_file")), "Uses an already opened file as input")
    .def("close", &Torch::File::close, (arg("self")), "Closes an opened file if it is opened. It is not an error to close the same file twice.")
    .def("isOpened", &Torch::File::isOpened, (arg("self")), "Says if the file pointed by this object is currently opened") 
    .def("eof", &Torch::File::eof, arg("self"), "Says if the file pointed by this object has already reached its end")
    .def("flush", &Torch::File::flush, arg("self"), "Flushes buffered output to disk immediatly")
    .def("seek", &Torch::File::seek, arg("self"), "Moves the internal position pointer to a different position")
    .def("tell", &Torch::File::tell, arg("self"), "Reports the current position to be read next")
    ;
}
