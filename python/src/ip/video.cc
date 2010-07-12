/**
 * @file src/ip/video.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Video framework to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "ip/Image.h"
#include "ip/Video.h"

using namespace boost::python;

void bind_ip_video()
{
  enum_<Torch::Video::State>("State")
    .value("Idle", Torch::Video::Idle)
    .value("Read", Torch::Video::Read)
    .value("Write", Torch::Video::Write)
    ;
  class_<Torch::Video, bases<Torch::Object> >("Video", 
      init<optional<const char*, const char*> >())
    .def("close", &Torch::Video::close)
    .def("read", &Torch::Video::read)
    .def("write", &Torch::Video::write)
    .def("codec", &Torch::Video::codec)
    .def("getNFrames", &Torch::Video::getNFrames)
    .def("getState", &Torch::Video::getState)
    ;
}
