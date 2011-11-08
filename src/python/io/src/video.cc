/**
 * @file python/databse/src/video.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds Video constructions to python 
 */

#include <boost/python.hpp>
#include <boost/python/slice.hpp>

#include "io/Video.h"
#include "core/python/exception.h"

#include "io/python/pyio.h"

using namespace boost::python;
namespace io = Torch::io;
namespace pyc = Torch::core::python;
namespace tp = Torch::python;

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

    tp::npyarray retval(reader->frame_type());
    o.read(retval); //note that this will advance the iterator
    return tp::npyarray_object(retval);
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
    PYTHON_ERROR(IndexError, "invalid index");
  }

  tp::npyarray retval(v.frame_type());
  io::VideoReader::const_iterator it = v.begin();
  it += frame;
  it.read(retval);
  return tp::npyarray_object(retval);
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
    PYTHON_ERROR(IndexError, "invalid start");
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
  for (size_t i=start, j=0; i<stop; i+=step, ++j, it+=(step-1)) {
    tp::npyarray tmp(v.frame_type());
    it.read(tmp);
    retval.append(tp::npyarray_object(tmp));
  }
 
  return tuple(retval);
}

static object videoreader_load(io::VideoReader& reader) {
  tp::npyarray tmp(reader.video_type());
  reader.load(tmp);
  return tp::npyarray_object(tmp);
}

static void videowriter_append(io::VideoWriter& writer, object a) {
  handle<> hdl((PyObject*)tp::describe_eltype(writer.video_type().dtype));
  object dtype(hdl);
  tp::npyarray tmp(a, dtype);
  writer.append(tmp);
}

void bind_io_video() {
  //special exceptions for videos
  pyc::register_exception_translator<Torch::io::VideoIsClosed>(PyExc_IOError);

  iterator_wrapper().wrap(); //wraps io::VideoReader::const_iterator

  class_<io::VideoReader, boost::shared_ptr<io::VideoReader> >("VideoReader",
      "VideoReader objects can read data from video files. The current implementation uses FFMPEG which is a stable freely available implementation for these tasks. You can read an entire video in memory by using the 'load()' method or use video iterators to read frame-by-frame and avoid overloading your machine's memory. The maximum precision FFMPEG will output is a 24-bit (8-bit per band) representation of each pixel (32-bit with transparency when supported by Torch, which is not the case presently). So, the input of data using this class uses uint8_t as base element type. Output will be colored using the RGB standard, with each band varying between 0 and 255, with zero meaning pure black and 255, pure white (color).", init<const std::string&>((arg("filename")), "Initializes a new VideoReader object by giving the input file name to read"))
    .add_property("filename", make_function(&io::VideoReader::filename, return_value_policy<copy_const_reference>()))
    .add_property("height", &io::VideoReader::height)
    .add_property("width", &io::VideoReader::width)
    .add_property("numberOfFrames", &io::VideoReader::numberOfFrames)
    .def("__len__", &io::VideoReader::numberOfFrames)
    .add_property("duration", &io::VideoReader::duration)
    .add_property("codecName", make_function(&io::VideoReader::codecName, return_value_policy<copy_const_reference>()))
    .add_property("codecLongName", make_function(&io::VideoReader::codecLongName, return_value_policy<copy_const_reference>()))
    .add_property("frameRate", &io::VideoReader::frameRate)
    .add_property("info", make_function(&io::VideoReader::info, return_value_policy<copy_const_reference>()))
    .add_property("video_type", make_function(&io::VideoReader::video_type, return_value_policy<copy_const_reference>()), "Typing information to load all of the file at once")
    .add_property("frame_type", make_function(&io::VideoReader::frame_type, return_value_policy<copy_const_reference>()), "Typing information to load the file frame by frame.")
    .def("load", &videoreader_load, (arg("self")), "Loads all of the video stream in a numpy ndarray organized in this way: (frames, color-bands, height, width). I'll dynamically allocate the output array and return it to you.")
    .def("__iter__", &io::VideoReader::begin)
    .def("__getitem__", &videoreader_getitem)
    .def("__getitem__", &videoreader_getslice)
    ;

  class_<io::VideoWriter, boost::shared_ptr<io::VideoWriter>, boost::noncopyable>("VideoWriter",
     "Use objects of this class to create and write video files using FFMPEG.",
     init<const std::string&, size_t, size_t, optional<float, float, size_t> >((arg("filename"), arg("height"), arg("width"), arg("framerate")=25.f, arg("bitrate")=1500000.f, arg("gop")=12), "Creates a new output file given the input parameters. The codec to be used will be derived from the filename extension."))
    .add_property("filename", make_function(&io::VideoWriter::filename, return_value_policy<copy_const_reference>()))
    .add_property("height", &io::VideoWriter::height)
    .add_property("width", &io::VideoWriter::width)
    .add_property("numberOfFrames", &io::VideoWriter::numberOfFrames)
    .def("__len__", &io::VideoWriter::numberOfFrames)
    .add_property("duration", &io::VideoWriter::duration)
    .add_property("codecName", make_function(&io::VideoWriter::codecName, return_value_policy<copy_const_reference>()))
    .add_property("codecLongName", make_function(&io::VideoWriter::codecLongName, return_value_policy<copy_const_reference>()))
    .add_property("frameRate", &io::VideoWriter::frameRate)
    .add_property("bitRate", &io::VideoWriter::bitRate)
    .add_property("gop", &io::VideoWriter::gop)
    .add_property("info", &io::VideoWriter::info)
    .add_property("is_opened", &io::VideoWriter::is_opened)
    .def("close", &io::VideoWriter::close, "Closes the current video stream and forces writing the trailer. After this point the video is finalized and cannot be written to anymore.")
    .add_property("video_type", make_function(&io::VideoWriter::video_type, return_value_policy<copy_const_reference>()), "Typing information to load all of the file at once")
    .add_property("frame_type", make_function(&io::VideoWriter::frame_type, return_value_policy<copy_const_reference>()), "Typing information to load the file frame by frame.")
    .def("append", &videowriter_append, (arg("self"), arg("frame")), "Writes a new frame or set of frames to the file. The frame should be setup as a array with 3 dimensions organized in this way (RGB color-bands, height, width). Sets of frames should be setup as a 4D array in this way: (frame-number, RGB color-bands, height, width). WARNING: At present time we only support arrays that have C-style storages (if you pass reversed arrays or arrays with Fortran-style storage, the result is undefined).")
    ;
}
