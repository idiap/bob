/**
 * @file daq/python/all.cc
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include <string>

#include <boost/format.hpp>

#include "bob/core/python/gil.h"
#include "bob/daq/Camera.h"
#include "bob/daq/VideoReaderCamera.h"
#include "bob/daq/OutputWriter.h"
#include "bob/daq/BobOutputWriter.h"
#include "bob/daq/Controller.h"
#include "bob/daq/SimpleController.h"
#include "bob/daq/Display.h"
#include "bob/daq/QtDisplay.h"
#include "bob/daq/FaceLocalization.h"
#include "bob/daq/NullFaceLocalization.h"
#include "bob/daq/ConsoleDisplay.h"
#include "bob/daq/CaptureSystem.h"
#include "bob/daq/VisionerFaceLocalization.h"

#ifdef V4L2
#include "bob/daq/V4LCamera.h"
#endif

using namespace bob::daq;

static boost::python::object getSupportedCamPixFormats(Camera& cam) {
  std::vector<Camera::CamPixFormat> pfs;
  cam.getSupportedCamPixFormats(pfs);

  boost::python::list l;

  for(unsigned int i = 0; i < pfs.size(); i++) {
    l.append(pfs[i]);
  }

  return l;
}

static boost::python::object getSupportedFrameSizes(Camera& cam, Camera::CamPixFormat pf) {
  std::vector<Camera::FrameSize> vec;
  cam.getSupportedFrameSizes(pf, vec);

  boost::python::list l;

  for(unsigned int i = 0; i < vec.size(); i++) {
    l.append(vec[i]);
  }

  return l;
}

static boost::python::object getSupportedFrameIntervals(Camera& cam, Camera::CamPixFormat pf, Camera::FrameSize& fs) {
  std::vector<Camera::FrameInterval> vec;
  cam.getSupportedFrameIntervals(pf, fs, vec);

  boost::python::list l;

  for(unsigned int i = 0; i < vec.size(); i++) {
    l.append(vec[i]);
  }

  return l;
}

static std::string FrameSize__str__(Camera::FrameSize const &self) {
  return boost::str(boost::format("%dx%d") % self.width % self.height);
}

static std::string FrameInterval__str__(Camera::FrameInterval const &self) {
  return boost::str(boost::format("%d/%d") % self.numerator % self.denominator);
}

void CaptureSystem_start(CaptureSystem& cs) {
  bob::python::no_gil unlock;
  return cs.start();
}

void Display_start(bob::daq::Display& d) {
  bob::python::no_gil unlock;
  return d.start();
}

int Camera_start(Camera& c) {
  bob::python::no_gil unlock;
  return c.start();
}

bool FaceLocalization_start(FaceLocalization& c) {
  bob::python::no_gil unlock;
  return c.start();
}

void VisionerFaceLocalization_del(VisionerFaceLocalization& vfl) {
  bob::python::no_gil unlock;
  vfl.join();
}

void bind_daq_all() {
  using namespace boost::python;

  /// CaptureSystem
  class_<CaptureSystem, boost::noncopyable>("CaptureSystem",
      "CaptureSystem is the main class used to capture images from a Camera "
      "and save it in a video file. CaptureSystem also displays a GUI with "
      "useful information about the current capture (remaining time, face "
      "detection, ...)",
      init<boost::shared_ptr<Camera>, const char*>(
         (arg("camera"), arg("faceLocalizationModelPath")),
         "'camera' is properly initialized Camera (used to grab the images)\n"
         "'faceLocalizationModelPath' is path to the Visioner face localization "
         "model"))
    .def("start", &CaptureSystem_start,(arg("self")), "Start the capture system. This call is blocking.")
    .add_property("length", &CaptureSystem::getLength, &CaptureSystem::setLength, "Recording length in seconds (recording delay excluded)")
    .add_property("recording_delay",  &CaptureSystem::getRecordingDelay,  &CaptureSystem::setRecordingDelay, "Recording delay in seconds, i.e. amount of seconds before the recording begins")
    .add_property("output_name", &CaptureSystem::getOutputName, &CaptureSystem::setOutputName, "Output name")
    .add_property("output_dir", &CaptureSystem::getOutputDir, &CaptureSystem::setOutputDir, "Directory where captured images are written")
    .add_property("thumbnail", &CaptureSystem::getThumbnail, &CaptureSystem::setThumbnail, "Path to an image, displayed as thumbnail in the GUI")
    .add_property("fullscreen", &CaptureSystem::getFullScreen, &CaptureSystem::setFullScreen, "GUI should be fullscreen")
    .def("set_display_size", &CaptureSystem::setDisplaySize, (arg("self"), arg("width"), arg("height")), "GUI size (ignored if fullscreen)")
    .def("set_execute_on_start_recording", &CaptureSystem::setExecuteOnStartRecording, (arg("self"), arg("command")), "Shell command executed when the recording starts\n. Warning: The command blocks the GUI thread. You should execute time * consuming commands in a sub-shell (e.g. command params &)")
    .def("set_execute_on_stop_recording", &CaptureSystem::setExecuteOnStopRecording, (arg("self"), arg("command")), "Shell command executed when the recording stops.\n Warning: See setExecuteOnStartRecording()")
    .def("set_text", &CaptureSystem::setText, (arg("self"), arg("text")), "Custom text displayed in the GUI");
  
  /// Callbacks
  class_<ControllerCallback, boost::noncopyable>("ControllerCallback", "Callback provided by a Controller", no_init)
    .def("image_received", &ControllerCallback::imageReceived, (arg("self"), arg("image"), arg("status")), "Image received by the Controller.\n 'image': pixel array in RGB 24 format.\n 'status': information about the frame");

  class_<Stoppable, boost::noncopyable>("Stoppable", no_init)
    .def("stop", &Stoppable::stop);

  class_<Camera::CameraCallback, boost::noncopyable>("CameraCallback", "Callback provided by Camera", no_init)
    .def("imageReceived", &Camera::CameraCallback::imageReceived,
      (arg("self"), arg("image"), arg("pixel_format"), arg("width"), arg("height"), arg("stride"), arg("size"), arg("frame_nb"), arg("timestamp")),
         "Image received by the camera. The implementation should be as short as"
         "possible since the capture thread is blocked during this call.\n"
         "\n"
         "'image': image buffer\n"
         "'pixel_format': pixel format of the image. \n"
         "'width': image width. \n"
         "'height': image height. \n"
         "'stride': image stride. \n"
         "'size': total image size (usually height*stride). \n"
         "'frame_nb': frame number. \n"
         "'timestamp': frame timestamp (in seconds). \n");

  class_<FaceLocalizationCallback, boost::noncopyable>("FaceLocalizationCallback", no_init)
    .def("on_detection", &FaceLocalizationCallback::onDetection, (arg("boundingbox")));
    
  /// Cameras
  enum_<Camera::CamPixFormat>("PixelFormat")
    .value("YUYV", Camera::YUYV)
    .value("MJPEG", Camera::MJPEG)
    .value("RGB24", Camera::RGB24)
    .value("OTHER", Camera::OTHER);
    
  class_<Camera::FrameSize>("FrameSize", init<int, int>((arg("width"), arg("height"))))
    .def_readwrite("width", &Camera::FrameSize::width)
    .def_readwrite("height", &Camera::FrameSize::height)
    .def("__eq__", &Camera::FrameSize::operator==)
    .def("__str__", &FrameSize__str__);
    
    class_<Camera::FrameInterval>("FrameInterval", "Frame interval (frame rate). You can compute frames per second using: fps = numerator / denominator",
                                  init<int, int>())
    .def_readwrite("numerator", &Camera::FrameInterval::numerator)
    .def_readwrite("denominator", &Camera::FrameInterval::denominator)
    .def("__eq__", &Camera::FrameInterval::operator==)
    .def("__str__", &FrameInterval__str__);
    
  class_<Camera, bases<Stoppable>, boost::noncopyable>("Camera", "Camera is an abstract class which captures frames", no_init)
    .def("start", &Camera_start, (arg("self")))
    .def("add_camera_callback", &Camera::addCameraCallback, (arg("self"), arg("callback")))
    .def("remove_camera_callback", &Camera::removeCameraCallback, (arg("self"), arg("callback")))
    .def("get_supported_pixel_formats", &getSupportedCamPixFormats, (arg("self")), "Get the list of supported pixel formats")
    .def("get_supported_frame_sizes", &getSupportedFrameSizes, (arg("self"), arg("pixel_format")), "Get the list of supported frame sizes for a pixel format")
    .def("get_supported_frame_intervals", &getSupportedFrameIntervals, (arg("self"), arg("pixel_format"), arg("frame_size")), "Get the list of supported frame intervals for a pixel format and a frame size")
    .add_property("pixel_format", &Camera::getCamPixFormat, &Camera::setCamPixFormat)
    .add_property("frame_size", &Camera::getFrameSize, &Camera::setFrameSize)
    .add_property("frame_interval", &Camera::getFrameInterval, &Camera::setFrameInterval)
    .def("print_summary", &Camera::printSummary, (arg("self")), "Print information about the device");

#ifdef V4L2
  class_<V4LCamera, bases<Camera>, boost::noncopyable>("V4LCamera", "Capture frames using Video for Linux 2",
                                                       init<const char*>((arg("device")), "'device': path to the video device (e.g. \"/dev/video0\")"));
#endif
  
  class_<VideoReaderCamera, bases<Camera>, boost::noncopyable>("VideoReaderCamera", "Read a video file from a bob::io::VideoReader",
                                                               init<boost::shared_ptr<bob::io::VideoReader> >((arg("video_reader"))));


  /// OutputWriters
  class_<OutputWriter, boost::noncopyable>("OutputWriter", "OutputWriter is an abstract class which provides a way to write frames on the hard drive.", no_init)
    .def("open", &OutputWriter::open, (arg("self")))
    .def("close", &OutputWriter::close, (arg("self")))
    .def("writeFrame", &OutputWriter::writeFrame, (arg("self"), arg("image"), arg("frame_nb"), arg("timestamp")), "Write a frame on the hard drive. \n'image': pixels in RGB24 format. \n'frame_nb': frame number. \n'timestamp': frame timestamp in seconds.")
    .def("set_output_dir", &OutputWriter::setOutputDir, (arg("self"), arg("dir")), "Set the directory where we want to output")
    .def("set_output_name", &OutputWriter::setOutputName, (arg("self"), arg("name")), "Set the name used to identify the output files");

    class_<BobOutputWriter, bases<OutputWriter>, boost::noncopyable>("BobOutputWriter",
                                                                     "Write a video file using Bob. Two files are created: .avi (contains the video with a fixed fps) and .txt (contains the timestamps for each frame)");

  /// Controllers
    class_<Controller, bases<Camera::CameraCallback>, boost::noncopyable>("Controller",
                                                                          "Controller is an abstract class which controls the capture process. It is "
                                                                          "responsible to forward captured images to other classes, and have to convert "
                                                                          "it to RGB24 format if needed",
                                                                          no_init)
    .def("add_controller_callback", &Controller::addControllerCallback, (arg("self"), arg("callback")))
    .def("remove_controller_callback", &Controller::removeControllerCallback, (arg("self"), arg("callback")))
    .def("add_stoppable", &Controller::addStoppable, (arg("self"), arg("stoppable")), "Add classes that should be stopped in priority (i.e. before ControllerCallback classes)")
    .def("remove_stoppable", &Controller::removeStoppable, (arg("self"), arg("stoppable")))
    .add_property("recording_delay", &Controller::getRecordingDelay, &Controller::setRecordingDelay, "Recording delay in seconds, i.e. amount of seconds before the recording begins.")
    .add_property("length", &Controller::getLength, &Controller::setLength, "Recording length in seconds (recording delay excluded)")
    .def("set_output_writer", &Controller::setOutputWriter, (arg("self"), arg("output_writer")));

    class_<SimpleController, bases<Controller>, boost::noncopyable>("SimpleController", "Default Controller implementation");

  /// Displays
    class_<bob::daq::Display, bases<ControllerCallback, FaceLocalizationCallback>, boost::noncopyable>("Display", "Display is an abstract class which is responsible to display an interface to the user", no_init)
    .def("start", &Display_start, (arg("self")), "Start the interface. This call should be blocking")
    .def("stop", &bob::daq::Display::stop, (arg("self")), "")
    .def("add_key_press_callback", &bob::daq::Display::addKeyPressCallback, (arg("self"), arg("callback")), "Add a callback which listen to user keyboard interactions.")
    .def("remove_key_press_callback", &bob::daq::Display::removeKeyPressCallback, (arg("self"), arg("callback")))
    .def("set_thumbnail", &bob::daq::Display::setThumbnail, (arg("self"), arg("path")), "Set path to an image, displayed as thumbnail in the GUI")
    .def("set_fullscreen", &bob::daq::Display::setFullscreen, (arg("self"), arg("fullscreen")), "Set whether GUI should be fullscreen")
    .def("set_display_size", &bob::daq::Display::setDisplaySize, (arg("self"), arg("width"), arg("height")), "Set GUI size (ignored if fullscreen)")
    .def("set_execute_on_start_recording", &bob::daq::Display::setExecuteOnStartRecording, (arg("self"), arg("command")), "Set a shell command executed when the recording starts. \nWarning: The command blocks the GUI thread. You should execute time consuming commands in a sub-shell (e.g. \"command params &\")")
    .def("set_execute_on_stop_recording", &bob::daq::Display::setExecuteOnStopRecording, (arg("self"), arg("command")), "Set a shell command executed when the recording stops. \nWarning: See setExecuteOnStartRecording()")
    .def("set_text", &bob::daq::Display::setText, (arg("self"), arg("text")), "Set custom text displayed in the GUI");

  class_<QtDisplay, bases<bob::daq::Display>, boost::noncopyable>("QtDisplay", "Display a GUI using Qt");

  class_<ConsoleDisplay, bases<bob::daq::Display>, boost::noncopyable>("ConsoleDisplay", "Dispay class that prints a console message when a frame or a detection is received.");


  /// FaceLocalizations
  class_<FaceLocalization, bases<ControllerCallback>, boost::noncopyable>("FaceLocalization", "FaceLocalization is an abstract class which provides face localization", no_init)
    .def("start", &FaceLocalization_start, (arg("self")), "Start the face localization of incoming frames")
    .def("add_face_localization_callback", &FaceLocalization::addFaceLocalizationCallback, (arg("self"), arg("callback")))
    .def("remove_face_localization_callback", &FaceLocalization::removeFaceLocalizationCallback, (arg("self"), arg("callback")));

  class_<VisionerFaceLocalization, bases<FaceLocalization>, boost::noncopyable>("VisionerFaceLocalization", "Provide face localization using Visioner",
                                                                                init<const char*>((arg("model_path")), "'model_path': path to a model file"))
    .def("__del__", &VisionerFaceLocalization_del, (arg("self")), "Destroys the face localization framework")
    ;
  
  class_<NullFaceLocalization, bases<FaceLocalization>, boost::noncopyable>("NullFaceLocalization", "NullFaceLocalization is an FaceLocalization which does nothing");

}
