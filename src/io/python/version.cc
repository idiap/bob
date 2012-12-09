/**
 * @file io/python/version.cc
 * @date Tue Nov 29 14:11:41 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Describes ways to retrieve version information about all dependent
 * packages.
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

#include "bob/config.h"

#include <boost/python.hpp>
#include <boost/format.hpp>
#include <cstdlib>
#include <string>

extern "C" {
#include <hdf5.h>
#include <jpeglib.h>

#define PNG_SKIP_SETJMP_CHECK
// #define requires because of the problematic pngconf.h.
// Look at the thread here: 
// https://bugs.launchpad.net/ubuntu/+source/libpng/+bug/218409
#include <png.h>

#if defined(HAVE_FFMPEG)
#  include <libavformat/avformat.h>
#  include <libavcodec/avcodec.h>
#  include <libavutil/avutil.h>
#  include <libswscale/swscale.h>
#endif 

#include <gif_lib.h>

#ifdef HAVE_MATIO
#include <matio.h>
#endif

#include <tiffio.h>
}

#if defined(HAVE_FFMPEG)
#include "bob/io/VideoUtilities.h"
#endif

using namespace boost::python;

/**
 * Version of HDF5 support
 */
static str hdf5_version() {
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(H5_VERS_MAJOR);
  f % BOOST_PP_STRINGIZE(H5_VERS_MINOR);
  f % BOOST_PP_STRINGIZE(H5_VERS_RELEASE);
  return str(f.str());
}

/**
 * FFmpeg version
 */
static dict ffmpeg_version() {
  dict v;
#if defined(HAVE_FFMPEG)
# if defined(FFMPEG_VERSION)
  if (std::strlen(FFMPEG_VERSION)) v["ffmpeg"] = str(FFMPEG_VERSION);
# endif
  v["avformat"] = str(BOOST_PP_STRINGIZE(LIBAVFORMAT_VERSION));
  v["avcodec"] = str(BOOST_PP_STRINGIZE(LIBAVFORMAT_VERSION));
  v["avutil"] = str(BOOST_PP_STRINGIZE(LIBAVUTIL_VERSION));
  v["swscale"] = str(BOOST_PP_STRINGIZE(LIBSWSCALE_VERSION));

  boost::format f("%d.%d.%d");
  f % ((0xff0000 & FFMPEG_VERSION_INT) >> 16)
    % ((0xff00 & FFMPEG_VERSION_INT) >> 8)
    % (0xff & FFMPEG_VERSION_INT);
  v["estimated"] = str(f.str());
#else
  v["ffmpeg"] = str("unavailable");
#endif
  return v;
}

/**
 * LibJPEG version
 */
static str libjpeg_version() {
  boost::format f("%d (compiled with %d bits depth)");
  f % JPEG_LIB_VERSION;
  f % BITS_IN_JSAMPLE;
  return str(f.str());
}

/**
 * Libpng version
 */
static str libpng_version() {
  return str(PNG_LIBPNG_VER_STRING);
}

/**
 * Libtiff version
 */
static str libtiff_version() {
  static const std::string beg_str("LIBTIFF, Version ");
  static const size_t beg_len = beg_str.size();
  std::string vtiff(TIFFGetVersion());

  // Remove first part if it starts with "LIBTIFF, Version "
  if(vtiff.compare(0, beg_len, beg_str) == 0)
    vtiff = vtiff.substr(beg_len); 

  // Remove multiple (copyright) lines if any
  size_t end_line = vtiff.find("\n");
  if(end_line != std::string::npos)
    vtiff = vtiff.substr(0,end_line); 

  return str(vtiff);
}

/**
 * Version of giflib support
 */
static str giflib_version() {
  str v;
#ifdef GIF_LIB_VERSION
 v = str(GIF_LIB_VERSION);
#else
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(GIFLIB_MAJOR);
  f % BOOST_PP_STRINGIZE(GIFLIB_MINOR);
  f % BOOST_PP_STRINGIZE(GIFLIB_RELEASE);
  v = str(f.str());
#endif
  return v;
}


/**
 * Matio, if compiled with such support
 */
static str matio_version() {
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

void bind_io_version() {
  dict vdict;
  vdict["HDF5"] = hdf5_version();
  vdict["FFmpeg"] = ffmpeg_version();
  vdict["libjpeg"] = libjpeg_version();
  vdict["libnetpbm"] = str("Unknown version");
  vdict["libpng"] = libpng_version();
  vdict["libtiff"] = libtiff_version();
  vdict["giflib"] = giflib_version();
  vdict["MatIO"] = matio_version();
  scope().attr("version") = vdict;
}
