/**
 * @file io/python/video.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds Video constructions to python
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
#include <boost/python/slice.hpp>

#include "bob/io/Video.h"
#include "bob/io/VideoUtilities.h"
#include "bob/core/python/exception.h"
#include "bob/core/python/ndarray.h"

using namespace boost::python;
namespace io = bob::io;
namespace tp = bob::python;

/**
 * Helper method for the the iterator wrapping
 */
static inline object pass_through(object const& o) { return o; }

/**
 * Python wrapper to make the Video::Reader::const_iterator look like a python
 * iterator
 */
struct iterator_wrapper {

  /**
   * Method to return the value pointed by the iterator and advance it
   */
  static object next (io::VideoReader::const_iterator& o) {
    const io::VideoReader* reader = o.parent();

    if (!reader) { //stop iteration immediately
      PYTHON_ERROR(StopIteration, "no more data");
    }

    //load the next frame: if an error is detected internally, throw
    tp::py_array retval(reader->frame_type());
    bool ok = o.read(retval); //note that this will advance the iterator
    if (!ok) PYTHON_ERROR(StopIteration, "iteration finished");
    return retval.pyobject();
  }

  /**
   * Method to wrap the iterator for python
   */
  static void wrap () {
    class_<io::VideoReader::const_iterator>("VideoReaderIterator", no_init)
      .def("next", next)
      .def("__iter__", pass_through)
      ;
  }

};

/**
 * Python wrapper to read a single frame from a video sequence, allowing the
 * implementation of a __getitem__() functionality on VideoReader objects.
 */
static object videoreader_getitem (io::VideoReader& v, Py_ssize_t sframe) {
  size_t frame = sframe;
  if (sframe < 0) frame = v.numberOfFrames() + sframe;

  if (frame >= v.numberOfFrames()) { //basic check
    PYTHON_ERROR(IndexError, "invalid index (" SIZE_T_FMT ") >= number of frames (" SIZE_T_FMT ")", frame, v.numberOfFrames());
  }

  tp::py_array retval(v.frame_type());
  io::VideoReader::const_iterator it = v.begin();
  it += frame;
  it.read(retval); //read and throw if a problem occurs
  return retval.pyobject();
}

/**
 * Python wrapper to read multiple frames from a video sequence, allowing the
 * implementation of a __getitem__() functionality on VideoReader objects.
 */
static tuple videoreader_getslice (io::VideoReader& v, slice sobj) {
  size_t start = 0;
  PySliceObject* sl = (PySliceObject*)sobj.ptr();
  if (sl->start != Py_None) {
    Py_ssize_t sstart = PyInt_AsLong(sl->start);
    start = sstart;
    if (sstart < 0) start = v.numberOfFrames() + sstart;
  }

  if (start >= v.numberOfFrames()) { //basic check
    PYTHON_ERROR(IndexError, "invalid start (" SIZE_T_FMT ") >= number of frames (" SIZE_T_FMT ")", start, v.numberOfFrames());
  }

  //the stop value may be None
  size_t stop = v.numberOfFrames();
  if (sl->stop != Py_None) {
    Py_ssize_t sstop = PyInt_AsLong(sl->stop);
    stop = sstop;
    if (sstop < 0) stop = v.numberOfFrames() + sstop;
  }
  if (stop >= v.numberOfFrames()) stop = v.numberOfFrames()+1;

  //the step value may be None
  int64_t step = 1;
  if (sl->step != Py_None) {
    step = PyInt_AsLong(sl->step);
  }

  //length of the sequence
  int length = (stop-start)/step;
  if (length == 0) length = 1; //a single return

  list retval;
  io::VideoReader::const_iterator it = v.begin();
  it += start;
  for (size_t i=start; it.parent() && i<stop; i+=step, it+=(step-1)) {
    tp::py_array tmp(v.frame_type());
    it.read(tmp); //throw if a problem occurs while reading the video
    retval.append(tmp.pyobject());
  }
 
  return tuple(retval);
}

static object videoreader_load(io::VideoReader& reader, 
  bool raise_on_error=false) {
  tp::py_array tmp(reader.video_type());
  size_t frames_read = reader.load(tmp, raise_on_error);
  return make_tuple(frames_read, tmp.pyobject());
}

BOOST_PYTHON_FUNCTION_OVERLOADS(videoreader_load_overloads, videoreader_load, 1, 2)

static void videowriter_append(io::VideoWriter& writer, object a) {
  tp::convert_t result = tp::convertible_to(a, writer.frame_type(),
      false, true);
  if (result != tp::IMPOSSIBLE) {
    tp::dtype dtype(writer.frame_type().dtype);
    tp::py_array tmp(a, dtype.self());
    writer.append(tmp);
  }
  else {
    tp::dtype dtype(writer.video_type().dtype);
    tp::py_array tmp(a, dtype.self());
    writer.append(tmp);
  }
}

/**
 * Describes a given codec or returns an empty dictionary, in case the codec
 * cannot be accessed
 */
static object describe_codec(const AVCodec* codec) {

  dict retval;

  retval["name"] = codec->name;
  retval["long_name"] = codec->long_name;
  retval["id"] = (unsigned)codec->id;

  // get specific framerates for the codec, if any:
  const AVRational* rate = codec->supported_framerates;
  list rates;
  while (rate && rate->num && rate->den) {
    rates.append( ((double)rate->num)/((double)rate->den) );
    ++rate;
  }
  retval["specific_framerates_hz"] = tuple(rates);

  // get codec capabilities
# ifdef CODEC_CAP_LOSSLESS
  retval["lossless"] = (bool)(codec->capabilities & CODEC_CAP_LOSSLESS);
# endif
# ifdef CODEC_CAP_EXPERIMENTAL
  retval["experimental"] = (bool)(codec->capabilities & CODEC_CAP_EXPERIMENTAL);
# endif
# ifdef CODEC_CAP_DELAY
  retval["delay"] = (bool)(codec->capabilities & CODEC_CAP_DELAY);
# endif
# ifdef CODEC_CAP_HWACCEL
  retval["hardware_accelerated"] = (bool)(codec->capabilities & CODEC_CAP_HWACCEL);
# endif
  retval["encode"] = (bool)(avcodec_find_encoder(codec->id));
  retval["decode"] = (bool)(avcodec_find_decoder(codec->id));
  
  return retval;
}

/**
 * Describes a given codec or returns an empty dictionary, in case the codec
 * cannot be accessed
 */
static object describe_encoder_by_name(const char* name) {
  AVCodec* codec = avcodec_find_encoder_by_name(name);
  if (!codec) return object();
  return describe_codec(codec);
}

/**
 * Describes a given codec or returns an empty dictionary, in case the codec
 * cannot be accessed
 */
static object describe_decoder_by_name(const char* name) {
  AVCodec* codec = avcodec_find_decoder_by_name(name);
  if (!codec) return object();
  return describe_codec(codec);
}

/**
 * Returns all output formats supported, related codecs and extensions
 */
static dict oformat_dictionary() {
  std::map<std::string, AVOutputFormat*> m;
  io::detail::ffmpeg::oformats_installed(m);
  dict retval;

  for (auto k=m.begin(); k!=m.end(); ++k) {
    dict property;
    property["name"] = k->second->name;
    property["long_name"] = k->second->long_name;
    property["mime_type"] = k->second->mime_type;

    // get extensions
    std::vector<std::string> exts;
    io::detail::ffmpeg::tokenize_csv(k->second->extensions, exts);
    list ext_list;
    for (auto ext=exts.begin(); ext!=exts.end(); ++ext) ext_list.append(*ext);
    property["extensions"] = tuple(ext_list);

    // get recommended codec
    if (!k->second->video_codec) {
      property["default_codec"] = object();
    }
    else {
      AVCodec* codec = avcodec_find_encoder(k->second->video_codec);
      if (!codec) property["default_codec"] = object();
      else property["default_codec"] = describe_codec(codec);
    }

    retval[k->first] = property;
  }

  return retval;
}

/**
 * Returns a dictionary of available codecs
 */
static object codec_dictionary() {
  std::map<std::string, const AVCodec*> m;
  io::detail::ffmpeg::codecs_installed(m);
  dict retval;
  for (auto k=m.begin(); k!=m.end(); ++k) {
    retval[k->first] = describe_codec(k->second);
  }
  return retval;
}

void bind_io_video() {
  //special exceptions for videos
  tp::register_exception_translator<bob::io::VideoIsClosed>(PyExc_IOError);

  iterator_wrapper().wrap(); //wraps io::VideoReader::const_iterator

  class_<io::VideoReader, boost::shared_ptr<io::VideoReader> >("VideoReader",
      "VideoReader objects can read data from video files. The current implementation uses FFMPEG which is a stable freely available implementation for these tasks. You can read an entire video in memory by using the 'load()' method or use video iterators to read frame-by-frame and avoid overloading your machine's memory. The maximum precision FFMPEG will output is a 24-bit (8-bit per band) representation of each pixel (32-bit with transparency when supported by bob, which is not the case presently). So, the input of data using this class uses uint8_t as base element type. Output will be colored using the RGB standard, with each band varying between 0 and 255, with zero meaning pure black and 255, pure white (color).", init<const std::string&>((arg("filename")), "Initializes a new VideoReader object by giving the input file name to read"))
    .add_property("filename", make_function(&io::VideoReader::filename, return_value_policy<copy_const_reference>()))
    .add_property("height", &io::VideoReader::height)
    .add_property("width", &io::VideoReader::width)
    .add_property("number_of_frames", &io::VideoReader::numberOfFrames)
    .def("__len__", &io::VideoReader::numberOfFrames)
    .add_property("duration", &io::VideoReader::duration)
    .add_property("codec_name", make_function(&io::VideoReader::codecName, return_value_policy<copy_const_reference>()))
    .add_property("codec_long_name", make_function(&io::VideoReader::codecLongName, return_value_policy<copy_const_reference>()))
    .add_property("frame_rate", &io::VideoReader::frameRate)
    .add_property("info", make_function(&io::VideoReader::info, return_value_policy<copy_const_reference>()))
    .add_property("video_type", make_function(&io::VideoReader::video_type, return_value_policy<copy_const_reference>()), "Typing information to load all of the file at once")
    .add_property("frame_type", make_function(&io::VideoReader::frame_type, return_value_policy<copy_const_reference>()), "Typing information to load the file frame by frame.")
    .def("__load__", &videoreader_load, videoreader_load_overloads((arg("self"), arg("raise_on_error")=false), "Loads all of the video stream in a numpy ndarray organized in this way: (frames, color-bands, height, width). I'll dynamically allocate the output array and return it to you. The flag 'raise_on_error', which is set to 'False' by default influences the error reporting in case problems are found with the video file. If you set it to 'True', we will report problems raising exceptions. If you either don't set it or set it to 'False', we will truncate the file at the frame with problems and will not report anything. It is your task to verify if the number of frames returned matches the expected number of frames as reported by the property 'numberOfFrames' in this object."))
    .def("__iter__", &io::VideoReader::begin, with_custodian_and_ward_postcall<0,1>())
    .def("__getitem__", &videoreader_getitem)
    .def("__getitem__", &videoreader_getslice)
    ;

  class_<io::VideoWriter, boost::shared_ptr<io::VideoWriter>, boost::noncopyable>("VideoWriter",
     "Use objects of this class to create and write video files using FFMPEG.",
#ifdef BOB_IO_VIDEOWRITER2_H
     init<const std::string&, size_t, size_t, optional<float, float, size_t, const std::string&, const std::string&> >((arg("filename"), arg("height"), arg("width"), arg("framerate")=25.f, arg("bitrate")=1500000.f, arg("gop")=12, arg("codec")="", arg("format")=""), "Creates a new output file given the input parameters. The format and codec to be used will be derived from the filename extension unless you define them explicetly (you can set both or just one of these two optional parameters)")
#else
     init<const std::string&, size_t, size_t, optional<float, float, size_t> >((arg("filename"), arg("height"), arg("width"), arg("framerate")=25.f, arg("bitrate")=1500000.f, arg("gop")=12), "Creates a new output file given the input parameters. The format and codec to be used will be derived from the filename extension unless you define them explicetly")
#endif
     )
#ifdef BOB_IO_VIDEOWRITER2_H
    .add_property("filename", &io::VideoWriter::filename)
#else
    .add_property("filename", make_function(&io::VideoWriter::filename, return_value_policy<copy_const_reference>()))
#endif
    .add_property("height", &io::VideoWriter::height)
    .add_property("width", &io::VideoWriter::width)
    .add_property("number_of_frames", &io::VideoWriter::numberOfFrames)
    .def("__len__", &io::VideoWriter::numberOfFrames)
    .add_property("duration", &io::VideoWriter::duration)
#ifdef BOB_IO_VIDEOWRITER2_H
    .add_property("format_name", &io::VideoWriter::formatName)
    .add_property("format_long_name", &io::VideoWriter::formatLongName)
    .add_property("codec_name", &io::VideoWriter::codecName)
    .add_property("codec_long_name", &io::VideoWriter::codecLongName)
#else
    .add_property("codec_name", make_function(&io::VideoWriter::codecName, return_value_policy<copy_const_reference>()))
    .add_property("codec_long_name", make_function(&io::VideoWriter::codecLongName, return_value_policy<copy_const_reference>()))
#endif
    .add_property("frame_rate", &io::VideoWriter::frameRate)
    .add_property("bit_rate", &io::VideoWriter::bitRate)
    .add_property("gop", &io::VideoWriter::gop)
    .add_property("info", &io::VideoWriter::info)
    .add_property("is_opened", &io::VideoWriter::is_opened)
    .def("close", &io::VideoWriter::close, "Closes the current video stream and forces writing the trailer. After this point the video is finalized and cannot be written to anymore.")
    .add_property("video_type", make_function(&io::VideoWriter::video_type, return_value_policy<copy_const_reference>()), "Typing information to load all of the file at once")
    .add_property("frame_type", make_function(&io::VideoWriter::frame_type, return_value_policy<copy_const_reference>()), "Typing information to load the file frame by frame.")
    .def("append", &videowriter_append, (arg("self"), arg("frame")), "Writes a new frame or set of frames to the file. The frame should be setup as a array with 3 dimensions organized in this way (RGB color-bands, height, width). Sets of frames should be setup as a 4D array in this way: (frame-number, RGB color-bands, height, width). WARNING: At present time we only support arrays that have C-style storages (if you pass reversed arrays or arrays with Fortran-style storage, the result is undefined).")
    ;

  def("video_codecs", &codec_dictionary, "Returns a dictionary containing a detailed description of the built-in codecs for videos");
  def("describe_video_encoder", &describe_encoder_by_name, "Describes a given video encoder (codec) starting with a name");
  def("describe_video_decoder", &describe_decoder_by_name, "Describes a given video decoder (codec) starting with a name");
  def("videowriter_formats", &oformat_dictionary, "Returns a dictionary containing a detailed description of the built-in output formats and default encoders for videos");
}
