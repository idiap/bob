/**
 * @file src/python/ip/src/videotensor.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the VideoTensor Framework to python
 */

#include <boost/python.hpp>

#include "ip/VideoTensor.h"

using namespace boost::python;

static int vt_width(const Torch::VideoTensor& vt) { return vt.size(1); }
static int vt_height(const Torch::VideoTensor& vt) { return vt.size(0); }
static int vt_color_planes(const Torch::VideoTensor& vt) { return vt.size(2); }
static int vt_frames(const Torch::VideoTensor& vt) { return vt.size(3); }

void bind_ip_videotensor()
{
  class_<Torch::VideoTensor, bases<Torch::ShortTensor> >("VideoTensor", " A video tensor is a representation of a video sequence as a Torch::Tensor. We follow the same implementation strategy as for Torch::Image, choosing a Torch::ShortTensor as a representation for the image sequence. We add a few methods to render the handling of videos and images a bit more pleasant to the end-user.", no_init)
    .def(init<Torch::Video&, optional<int> >((arg("video"), arg("color_planes")=3), "Builds a new VideoTensor from an existing Video object. You can cause RGB to gray conversion if you specify color_planes=1. Please note any other value other than '1' or '3' for this parameter will not work."))
    .def(init<Torch::VideoTensor&>((arg("other")), "Copies the data from another VideoTensor"))
    .def(init<Torch::TensorFile&>((arg("tensor_file")), "Builds a new VideoTensor from data existing in a TensorFile. This method will read the next available tensor in the file."))
    .def(init<int, int, int, int>((arg("width"), arg("height"), arg("color_planes"), arg("frames")), "Builds a new VideoTensor specifying the size, number of color planes and total number of frames"))
    .def("getFrame", &Torch::VideoTensor::getFrame, (arg("self"), arg("image"), arg("frame")), "Sets the image object to the i-th image on the sequence (this will set by reference, so it is fast!)")
    .def("setFrame", &Torch::VideoTensor::setFrame, (arg("self"), arg("image"), arg("frame")), "Resets a certain image in the video sequence to the value given as input. Please note that the image specifications (width and height) should respect the values in the video tensor. If the number of planes varies between this and the source image, the adaptation also found in Image::copyFrom() will be used.") 
    .def("save", (bool (Torch::VideoTensor::*)(Torch::Video&) const)&Torch::VideoTensor::save, (arg("self"), arg("video")), "Saves the current video tensor in a Video file") 
    .def("save", (bool (Torch::VideoTensor::*)(Torch::TensorFile&) const)&Torch::VideoTensor::save, (arg("self"), arg("video")), "Saves the current video tensor in a already opened TensorFile") 
    .add_property("width", &vt_width)
    .add_property("height", &vt_height)
    .add_property("planes", &vt_color_planes)
    .add_property("frames", &vt_frames)
    ;
}
