/**
 * @file src/ip/video.cc
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

static int get_video_height(Torch::Video& v) 
{ return v.getIOption("height"); }

static float get_video_bitrate(Torch::Video& v) 
{ return v.getFOption("bitrate"); }

static float get_video_framerate(Torch::Video& v) 
{ return v.getFOption("framerate"); }

static int get_video_gop(Torch::Video& v) 
{ return v.getIOption("gop"); }

static bool get_video_verbosity(Torch::Video& v) 
{ return v.getBOption("verbose"); }

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

void bind_ip_video()
{
  enum_<Torch::Video::State>("State")
    .value("Idle", Torch::Video::Idle)
    .value("Read", Torch::Video::Read)
    .value("Write", Torch::Video::Write)
    ;
  class_<Torch::Video, boost::shared_ptr<Torch::Video>, bases<Torch::Object> >("Video", 
      init<optional<const char*, const char*> >())
    .def("__init__", make_constructor(&make_readable_video))
    .def("__init__", make_constructor(&make_readable_video_quietly))
    .def("__init__", make_constructor(&make_writeable_video))
    .def("__init__", make_constructor(&make_writeable_video_quietly))
    .def("__init__", make_constructor(&make_writeable_video_like))
    .def("__init__", make_constructor(&make_writeable_video_like_quietly))
    .def("close", &Torch::Video::close)
    .def("read", &Torch::Video::read)
    .def("write", &Torch::Video::write)
    .def("codec", &Torch::Video::codec)
    .def("getNFrames", &Torch::Video::getNFrames)
    .def("getState", &Torch::Video::getState)
    .def("width", &get_video_width)
    .def("height", &get_video_height)
    .def("bitrate", &get_video_bitrate)
    .def("framerate", &get_video_framerate)
    .def("gop", &get_video_gop)
    ;
}
