/**
 * @file src/ip/video.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Video framework to python
 */

#include <boost/python.hpp>

#include "ip/VideoTensor.h"

using namespace boost::python;

void bind_ip_videotensor()
{
  class_<Torch::VideoTensor, bases<Torch::ShortTensor> >("VideoTensor", " A video tensor is a representation of a video sequence as a Torch::Tensor. We follow the same implementation strategy as for Torch::Image, choosing a Torch::ShortTensor as a representation for the image sequence. We add a few methods to render the handling of videos and images a bit more pleasant to the end-user.", no_init)
    .def(init<Torch::Video&>((arg("video")), "Builds a new VideoTensor from an existing Video object"))
    .def(init<Torch::VideoTensor&>((arg("other")), "Copies the data from another VideoTensor"))
    .def(init<Torch::TensorFile&>((arg("tensor_file")), "Builds a new VideoTensor from data existing in a TensorFile. This method will read the next available tensor in the file."))
    .def(init<int, int, int, int>((arg("width"), arg("height"), arg("color_planes"), arg("frames")), "Builds a new VideoTensor specifying the size, number of color planes and total number of frames"))
    .def("getFrame", &Torch::VideoTensor::getFrame, (arg("self"), arg("image"), arg("frame")), "Sets the image object to the i-th image on the sequence (this will set by reference, so it is fast!)")
    .def("setFrame", &Torch::VideoTensor::setFrame, (arg("self"), arg("image"), arg("frame")), "Resets a certain image in the video sequence to the value given as input. Please note that the image specifications (width and height) should respect the values in the video tensor. If the number of planes varies between this and the source image, the adaptation also found in Image::copyFrom() will be used.") 
    .def("save", (bool (Torch::VideoTensor::*)(Torch::Video&) const)&Torch::VideoTensor::save, (arg("self"), arg("video")), "Saves the current video tensor in a Video file") 
    .def("save", (bool (Torch::VideoTensor::*)(Torch::TensorFile&) const)&Torch::VideoTensor::save, (arg("self"), arg("video")), "Saves the current video tensor in a already opened TensorFile") 
    ;
}
