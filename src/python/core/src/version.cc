/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 27 Apr 19:21:25 2011 
 *
 * @brief Describes ways to retrieve version information about all dependent
 * packages.
 */

#include <boost/python.hpp>
#include <string>
#include <blitz/blitz.h>
#include <boost/version.hpp>
#include <boost/format.hpp>
#include <ImageMagick/Magick++.h> 

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <numpy/arrayobject.h>
#include <hdf5.h>
#ifdef HAVE_MATIO
#include <matio.h>
#endif
}

using namespace boost::python;

/**
 * Describes the blitz version and information
 */
str blitz_version() {
  std::string retval(BZ_VERSION);
# ifdef HAVE_BLITZ_SIZETYPE
  //this is a temporary hack to identify support for more than 2GB big arrays
  retval += "+CVS (>2GB contents supported)";
# endif
  return str(retval);
}

/**
 * Describes the version of Boost libraries installed
 */
str boost_version() {
  boost::format f("%d.%d.%d");
  f % (BOOST_VERSION / 100000);
  f % (BOOST_VERSION / 100 % 1000);
  f % (BOOST_VERSION % 100);
  return str(f.str());
}

/**
 * Describes the compiler version
 */
tuple compiler_version() {
# if defined(__GNUC__)
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(__GNUC__);
  f % BOOST_PP_STRINGIZE(__GNUC_MINOR__);
  f % BOOST_PP_STRINGIZE(__GNUC_PATCHLEVEL__);
  return make_tuple(str("gcc"), str(f.str()));
# elif defined(__llvm__)
  return make_tuple(str("llvm"), str(BOOST_PP_STRINGIZE(__llvm__)));
# elif defined(__clang__)
  boost::format f("clang %s.%s.%s");
  f % BOOST_PP_STRINGIZE(__clang_major__);
  f % BOOST_PP_STRINGIZE(__clang_minor__);
  f % BOOST_PP_STRINGIZE(__clang_patchlevel__);
  return make_tuple(str("clang"), str(f.str()));
# else
  return str("unsupported");
# endif
}

/**
 * Python version with which we compiled the extensions
 */
str python_version() {
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(PY_MAJOR_VERSION);
  f % BOOST_PP_STRINGIZE(PY_MINOR_VERSION);
  f % BOOST_PP_STRINGIZE(PY_MICRO_VERSION);
  return str(f.str());
}

/**
 * Version of HDF5 support
 */
str hdf5_version() {
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(H5_VERS_MAJOR);
  f % BOOST_PP_STRINGIZE(H5_VERS_MINOR);
  f % BOOST_PP_STRINGIZE(H5_VERS_RELEASE);
  return str(f.str());
}

/**
 * FFmpeg version
 */
tuple ffmpeg_version() {
  tuple avformat = make_tuple(str("avformat"), str(BOOST_PP_STRINGIZE(LIBAVFORMAT_VERSION)));
  tuple avcodec = make_tuple(str("avcodec"), str(BOOST_PP_STRINGIZE(LIBAVCODEC_VERSION)));
  tuple avutil = make_tuple(str("avutil"), str(BOOST_PP_STRINGIZE(LIBAVUTIL_VERSION)));
  tuple swscale = make_tuple(str("swscale"), str(BOOST_PP_STRINGIZE(LIBSWSCALE_VERSION)));
  return make_tuple(avformat, avcodec, avutil, swscale);
}

/**
 * ImageMagick version
 */
str magick_version() {
  return str(MagickLibVersionText);
}

/**
 * Numpy version
 */
str numpy_version() {
  return str(BOOST_PP_STRINGIZE(NPY_VERSION));
}

/**
 * Matio, if compiled with such support
 */
str matio_version() {
#ifdef HAVE_MATIO
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(MATIO_MAJOR_VERSION);
  f % BOOST_PP_STRINGIZE(MATIO_MINOR_VERSION);
  f % BOOST_PP_STRINGIZE(MATIO_RELEASE_LEVEL);
  return str(f.str());
#else
  return str("unavailable");
#endif
}

void bind_version_info() {
  def("blitz_version", &blitz_version);
  def("boost_version", &boost_version);
  def("compiler_version", &compiler_version);
  def("python_version", &python_version);
  def("hdf5_version", &hdf5_version);
  def("ffmpeg_version", &ffmpeg_version);
  def("magick_version", &magick_version);
  def("numpy_version", &numpy_version);
  def("matio_version", &matio_version);
}
