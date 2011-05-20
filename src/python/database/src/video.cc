/**
 * @file python/databse/src/video.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds Video constructions to python 
 */

#include <boost/python.hpp>
#include <boost/python/slice.hpp>

#include "database/Video.h"
#include "core/python/exception.h"

using namespace boost::python;
using namespace Torch::core::python;
namespace db = Torch::database;

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
  static blitz::Array<uint8_t,3> next (db::VideoReader::const_iterator& o) {
    const db::VideoReader* reader = o.parent();

    if (!reader) { //stop iteration immediately
      PyErr_SetString(PyExc_StopIteration, "No more data.");
      throw_error_already_set();
    }

    blitz::Array<uint8_t,3> retval(3, reader->height(), reader->width());
    o.read(retval); //note that this will advance the iterator
    return retval;
  }

  /**
   * Method to wrap the iterator for python
   */
  static void wrap () {
    class_<db::VideoReader::const_iterator>("VideoReaderIterator", no_init)
      .def("next", next)
      .def("__iter__", pass_through)
      ;
  }

};

/**
 * Python wrapper to read a single frame from a video sequence, allowing the
 * implementation of a __getitem__() functionality on VideoReader objects.
 */
static blitz::Array<uint8_t,3> videoreader_getitem (db::VideoReader& v, Py_ssize_t sframe) {
  size_t frame = sframe;
  if (sframe < 0) frame = v.numberOfFrames() + sframe;

  if (frame >= v.numberOfFrames()) { //basic check
    PyErr_SetString(PyExc_IndexError, "Invalid index");
    throw_error_already_set();
  }

  blitz::Array<uint8_t,3> retval(3, v.height(), v.width());
  db::VideoReader::const_iterator it = v.begin();
  it += frame;
  it.read(retval);
  return retval;
}

/**
 * Python wrapper to read multiple frames from a video sequence, allowing the
 * implementation of a __getitem__() functionality on VideoReader objects.
 */
static blitz::Array<uint8_t,4> videoreader_getslice (db::VideoReader& v, slice sobj) {
  size_t start = 0;
  PySliceObject* sl = (PySliceObject*)sobj.ptr();
  if (sl->start != Py_None) {
    Py_ssize_t sstart = PyInt_AsLong(sl->start);
    start = sstart;
    if (sstart < 0) start = v.numberOfFrames() + sstart;
  }

  if (start >= v.numberOfFrames()) { //basic check
    PyErr_SetString(PyExc_IndexError, "Invalid start");
    throw_error_already_set();
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

  blitz::Array<uint8_t,4> retval(length, 3, v.height(), v.width());
  db::VideoReader::const_iterator it = v.begin();
  it += start;
  blitz::Range a = blitz::Range::all();
  for (size_t i=start, j=0; i<stop; i+=step, ++j, it+=(step-1)) {
    blitz::Array<uint8_t,3> ref = retval(j, a, a, a);
    it.read(ref);
  }
  return retval;
}

void bind_database_video() {
  //exceptions for videos
  CxxToPythonTranslatorPar2<Torch::database::FFmpegException, Torch::database::Exception, const char*, const char*>("FFmpegException", "Thrown when there is a problem with a Video file.");

  iterator_wrapper().wrap(); //wraps db::VideoReader::const_iterator

  class_<db::VideoReader, boost::shared_ptr<db::VideoReader> >("VideoReader",
      "VideoReader objects can read data from video files. The current implementation uses FFMPEG which is a stable freely available implementation for these tasks. You can read an entire video in memory by using the 'load()' method or use video iterators to read frame-by-frame and avoid overloading your machine's memory. The maximum precision FFMPEG will output is a 24-bit (8-bit per band) representation of each pixel (32-bit with transparency when supported by Torch, which is not the case presently). So, the input of data using this class uses uint8_t as base element type. Output will be colored using the RGB standard, with each band varying between 0 and 255, with zero meaning pure black and 255, pure white (color).", init<const std::string&>((arg("filename")), "Initializes a new VideoReader object by giving the input file name to read"))
    .add_property("filename", make_function(&db::VideoReader::filename, return_value_policy<copy_const_reference>()))
    .add_property("height", &db::VideoReader::height)
    .add_property("width", &db::VideoReader::width)
    .add_property("numberOfFrames", &db::VideoReader::numberOfFrames)
    .def("__len__", &db::VideoReader::numberOfFrames)
    .add_property("duration", &db::VideoReader::duration)
    .add_property("codecName", make_function(&db::VideoReader::codecName, return_value_policy<copy_const_reference>()))
    .add_property("codecLongName", make_function(&db::VideoReader::codecLongName, return_value_policy<copy_const_reference>()))
    .add_property("frameRate", &db::VideoReader::frameRate)
    .add_property("info", make_function(&db::VideoReader::info, return_value_policy<copy_const_reference>()))
    .def("load", &db::VideoReader::load, (arg("array")), "Loads all of the video stream in a blitz array organized in this way: (frames, color-bands, height, width). The 'data' parameter will be resized if required.")
    .def("__iter__", &db::VideoReader::begin)
    .def("__getitem__", &videoreader_getitem)
    .def("__getitem__", &videoreader_getslice)
    ;

  class_<db::VideoWriter, boost::shared_ptr<db::VideoWriter>, boost::noncopyable>("VideoWriter",
     "Use objects of this class to create and write video files using FFMPEG.",
     init<const std::string&, size_t, size_t, optional<float, float, size_t> >((arg("filename"), arg("height"), arg("width"), arg("framerate")=25.f, arg("bitrate")=1500000.f, arg("gop")=12), "Creates a new output file given the input parameters. The codec to be used will be derived from the filename extension."))
    .add_property("filename", make_function(&db::VideoWriter::filename, return_value_policy<copy_const_reference>()))
    .add_property("height", &db::VideoWriter::height)
    .add_property("width", &db::VideoWriter::width)
    .add_property("numberOfFrames", &db::VideoWriter::numberOfFrames)
    .def("__len__", &db::VideoWriter::numberOfFrames)
    .add_property("duration", &db::VideoWriter::duration)
    .add_property("codecName", make_function(&db::VideoWriter::codecName, return_value_policy<copy_const_reference>()))
    .add_property("codecLongName", make_function(&db::VideoWriter::codecLongName, return_value_policy<copy_const_reference>()))
    .add_property("frameRate", &db::VideoWriter::frameRate)
    .add_property("bitRate", &db::VideoWriter::bitRate)
    .add_property("gop", &db::VideoWriter::gop)
    .add_property("info", &db::VideoWriter::info)
    .def("append", (void (db::VideoWriter::*)(const blitz::Array<uint8_t,3>&))&db::VideoWriter::append, (arg("frame")), "Writes a new frame to the file. The frame should be setup as a blitz::Array<> with 3 dimensions organized in this way (RGB color-bands, height, width). WARNING: At present time we only support arrays that have C-style storages (if you pass reversed arrays or arrays with Fortran-style storage, the result is undefined).")
    .def("append", (void (db::VideoWriter::*)(const blitz::Array<uint8_t,4>&))&db::VideoWriter::append, (arg("frame")), "Writes a set of frames to the file. The frame set should be setup as a blitz::Array<> with 4 dimensions organized in this way: (frame-number, RGB color-bands, height, width). WARNING: At present time we only support arrays that have C-style storages (if you pass reversed arrays or arrays with Fortran-style storage, the result is undefined).")
    ;
}
