/**
 * @file src/python/ip/src/video.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Video framework to python
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/Object.h"
#include "ip/Image.h"
#include "ip/Video.h"

using namespace boost::python;

/**
 * Helps creating an new output video file in the "certified way"...
 * @param name The name of the output video file
 * @param width The width of the output video
 * @param height The height of the output video
 * @param bitrate The bitrate of the output video
 * @param framerate The framerate of the output video
 * @param gop The group of pictures size for fast-forwarding and reversing
 * @param verbose Makes the implementation be a little bit more verbose
 *
 * @return A new video object that should be managed by the caller
 */
static boost::shared_ptr<Torch::Video> make_writeable_video(const char* name,
    int width, int height, float bitrate, float framerate, 
    int gop, bool verbose) {
  boost::shared_ptr<Torch::Video> retval(new Torch::Video());
  bool success = true;
	success &= retval->setIOption("width", width);
	success &= retval->setIOption("height", height);
  success &= retval->setFOption("bitrate", bitrate);
  success &= retval->setFOption("framerate", framerate);
	success &= retval->setIOption("gop", gop);
	success &= retval->setBOption("verbose", verbose);
  success &= retval->open(name, "w");
  if (success) return retval;
  return boost::shared_ptr<Torch::Video>();
}

static boost::shared_ptr<Torch::Video> 
make_writeable_video_quietly(const char* name,
    int width, int height, float bitrate, float framerate, int gop) {
  return make_writeable_video(name, width, height, bitrate, framerate, gop, 
      false);
}

/**
 * Helps creating a new input video out of a filename. This method was created
 * to avoid altogether direct use of the Video constructors, which do not work
 * correctly.
 * @param name The name of the input video
 * @param verbose Makes the implementation be a little bit more verbose
 * 
 * @return A new video object that should be managed by the caller
 */
static boost::shared_ptr<Torch::Video> make_readable_video(const char* name, 
    bool verbose)
{
  boost::shared_ptr<Torch::Video> retval(new Torch::Video());
  bool success = true;
	success &= retval->setBOption("verbose", verbose);
  success &= retval->open(name, "r");
  if (success) return retval;
  return boost::shared_ptr<Torch::Video>();
}


static boost::shared_ptr<Torch::Video> make_readable_video_quietly(const char* name)
{
  return make_readable_video(name, false);
}

/**
 * A few helpers to retrieve information from video files as recommended on the
 * ip/Video.h documentation
 */
static int get_video_width(Torch::Video& v) 
{ return v.getIOption("width"); }

static void set_video_width(Torch::Video& v, int val)
{ v.setIOption("width", val); }

static int get_video_height(Torch::Video& v) 
{ return v.getIOption("height"); }

static void set_video_height(Torch::Video& v, int val)
{ v.setIOption("height", val); }

static float get_video_bitrate(Torch::Video& v) 
{ return v.getFOption("bitrate"); }

static void set_video_bitrate(Torch::Video& v, float val)
{ v.setFOption("bitrate", val); }

static float get_video_framerate(Torch::Video& v) 
{ return v.getFOption("framerate"); }

static void set_video_framerate(Torch::Video& v, float val)
{ v.setFOption("framerate", val); }

static int get_video_gop(Torch::Video& v) 
{ return v.getIOption("gop"); }

static void set_video_gop(Torch::Video& v, int val)
{ v.setIOption("gop", val); }

static bool get_video_verbose(Torch::Video& v) 
{ return v.getBOption("verbose"); }

static void set_video_verbose(Torch::Video& v, bool val) 
{ v.setBOption("verbose", val); }

/**
 * A helper to make videos like another existing one, minus the name and
 * verbosity markings
 */
static boost::shared_ptr<Torch::Video> make_writeable_video_like
(const char* name, Torch::Video& v, bool verbose) 
{
  return make_writeable_video(name, get_video_width(v), get_video_height(v),
      get_video_bitrate(v), get_video_framerate(v), get_video_gop(v), verbose);
}

static boost::shared_ptr<Torch::Video> make_writeable_video_like_quietly
(const char* name, Torch::Video& v) 
{
  return make_writeable_video(name, get_video_width(v), get_video_height(v),
      get_video_bitrate(v), get_video_framerate(v), get_video_gop(v), false);
}

static bool video_write(Torch::Video& v, const Torch::Image& i) {
  if (i.getNPlanes() == 3) return v.write(i);

  //we are dealing with grayscale images, needs expansion to RGB
  Torch::Image color_i(i.getWidth(), i.getHeight(), 3);
  Torch::ShortTensor grays;
  grays.select(&i, 2, 0); //get the gray levels
  for (unsigned i=0; i<3; ++i) {
    Torch::ShortTensor s;
    s.select(&color_i, 2, i);
    s.copy(&grays); //copy the gray levels over the 3 dimensions
  }
  return v.write(color_i);
}

void bind_ip_video()
{
  enum_<Torch::Video::State>("State")
    .value("Idle", Torch::Video::Idle)
    .value("Read", Torch::Video::Read)
    .value("Write", Torch::Video::Write)
    ;
  class_<Torch::Video, boost::shared_ptr<Torch::Video>, bases<Torch::Object> >("Video", "Allows manipulation and creation of video files", 
      init<optional<const char*, const char*> >((arg("filename"), arg("mode"))))
    .def("__init__", make_constructor(&make_readable_video))
    .def("__init__", make_constructor(&make_readable_video_quietly))
    .def("__init__", make_constructor(&make_writeable_video))
    .def("__init__", make_constructor(&make_writeable_video_quietly))
    .def("__init__", make_constructor(&make_writeable_video_like))
    .def("__init__", make_constructor(&make_writeable_video_like_quietly))
    .def("close", &Torch::Video::close, arg("self"), "Closes the video file, if it was opened")
    .def("read", &Torch::Video::read, (arg("self"), arg("image")), "Reads a single image from the video file.\n\nThis method will copy the video frame pixmap into the image object given as input parameter.")
    .def("write", &video_write, (arg("self"), arg("image")), "Write an image file into the video stream (single frame)")
    .add_property("codec", &Torch::Video::codec)
    .add_property("nframes", &Torch::Video::getNFrames)
    .add_property("state", &Torch::Video::getState)
    .add_property("width", &get_video_width, &set_video_width)
    .add_property("height", &get_video_height, &set_video_height) 
    .add_property("bitrate", &get_video_bitrate, &set_video_bitrate)
    .add_property("framerate", &get_video_framerate, &set_video_framerate)
    .add_property("gop", &get_video_gop, &set_video_gop)
    .add_property("verbose", &get_video_verbose, &set_video_verbose)
    ;
}
