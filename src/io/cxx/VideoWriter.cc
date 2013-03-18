/**
 * @file io/cxx/VideoWriter.cc
 * @date Wed 28 Nov 2012 13:51:58 CET
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A class to help you write videos. This code originates from
 * http://ffmpeg.org/doxygen/1.0/, "muxing.c" example.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <boost/format.hpp>
#include <boost/preprocessor.hpp>
#include "bob/io/VideoWriter.h"

#if LIBAVFORMAT_VERSION_INT < 0x361764 /* 54.23.100 @ ffmpeg-0.11 */
#define FFMPEG_VIDEO_BUFFER_SIZE 200000
#else
#define FFMPEG_VIDEO_BUFFER_SIZE 0
#endif

#ifndef AV_PIX_FMT_RGB24
#define AV_PIX_FMT_RGB24 PIX_FMT_RGB24
#endif

namespace ffmpeg = bob::io::detail::ffmpeg;

bob::io::VideoWriter::VideoWriter(
    const std::string& filename,
    size_t height,
    size_t width,
    double framerate,
    double bitrate,
    size_t gop,
    const std::string& codec,
    const std::string& format) :
  m_filename(filename),
  m_opened(false),
  m_format_context(ffmpeg::make_output_format_context(filename, format)),
  m_codec(ffmpeg::find_encoder(filename, m_format_context, codec)),
  m_stream(ffmpeg::make_stream(filename, m_format_context, codec, height,
        width, framerate, bitrate, gop, m_codec)),
  m_codec_context(ffmpeg::make_codec_context(filename, m_stream.get(), m_codec)),
  m_context_frame(ffmpeg::make_frame(filename, m_codec_context, m_stream->codec->pix_fmt)),
#if LIBAVCODEC_VERSION_INT >= 0x352a00 //53.42.0 @ ffmpeg-0.9
  m_swscaler(ffmpeg::make_scaler(filename, m_codec_context, PIX_FMT_GBRP, m_stream->codec->pix_fmt)),
#else
  m_rgb24_frame(ffmpeg::make_frame(filename, m_codec_context, AV_PIX_FMT_RGB24)),
  m_swscaler(ffmpeg::make_scaler(filename, m_codec_context, AV_PIX_FMT_RGB24, m_stream->codec->pix_fmt)),
#endif
  m_buffer(ffmpeg::make_buffer(m_format_context, FFMPEG_VIDEO_BUFFER_SIZE)),
  m_height(height),
  m_width(width),
  m_framerate(framerate),
  m_bitrate(bitrate),
  m_gop(gop),
  m_codecname(codec),
  m_formatname(format),
  m_current_frame(0)
{
  ffmpeg::open_output_file(m_filename, m_format_context);

  //sets up the io layer typeinfo
  m_typeinfo_video.dtype = m_typeinfo_frame.dtype = bob::core::array::t_uint8;
  m_typeinfo_video.nd = 4;
  m_typeinfo_frame.nd = 4;
  m_typeinfo_video.shape[0] = 0;
  m_typeinfo_video.shape[1] = m_typeinfo_frame.shape[0] = 3;
  m_typeinfo_video.shape[2] = m_typeinfo_frame.shape[1] = height;
  m_typeinfo_video.shape[3] = m_typeinfo_frame.shape[2] = width;
  m_typeinfo_frame.update_strides();
  m_typeinfo_video.update_strides();

  //resets the output frame PTS [Note: presentation timestamp in time_base
  //units (time when frame should be shown to user) If AV_NOPTS_VALUE then
  //frame_rate = 1/time_base will be assumed].
  m_context_frame->pts = 0;

  m_opened = true; ///< file is now considered opened for bussiness
}

bob::io::VideoWriter::~VideoWriter() {
  if (m_opened) close();
}

void bob::io::VideoWriter::close() {

  ffmpeg::flush_encoder(m_filename, m_format_context, m_stream, m_codec,
      m_buffer, FFMPEG_VIDEO_BUFFER_SIZE);
  ffmpeg::close_output_file(m_filename, m_format_context);

  /* Destroyes resources in an orderly fashion */
  m_codec_context.reset();
  m_context_frame.reset();
  m_rgb24_frame.reset();
  m_buffer.reset();
  m_swscaler.reset();
  m_stream.reset();
  m_format_context.reset();

  m_opened = false; ///< file is now considered closed
}

std::string bob::io::VideoWriter::info() const {
  /**
   * This will create a local description of the contents of the stream, in
   * printable format.
   */
  boost::format info("Video file: %s; FFmpeg: avformat-%s; avcodec-%s; avutil-%s; swscale-%d; Format: %s (%s); Codec: %s (%s); Time: %.2f s (%d @ %2.fHz); Size (w x h): %d x %d pixels");
  info % m_filename;
  info % BOOST_PP_STRINGIZE(LIBAVFORMAT_VERSION);
  info % BOOST_PP_STRINGIZE(LIBAVCODEC_VERSION);
  info % BOOST_PP_STRINGIZE(LIBAVUTIL_VERSION);
  info % BOOST_PP_STRINGIZE(LIBSWSCALE_VERSION);
  info % m_format_context->oformat->name;
  info % m_format_context->oformat->long_name;
  info % m_stream->codec->codec->name;
  info % m_stream->codec->codec->long_name;
  info % (m_current_frame/m_framerate);
  info % m_current_frame;
  info % m_framerate;
  info % m_width;
  info % m_height;
  return info.str();
}

void bob::io::VideoWriter::append(const blitz::Array<uint8_t,4>& data) {
  if (!m_opened) {
    boost::format m("video writer for file `%s' is closed and cannot be written to");
    m % m_filename;
    throw std::runtime_error(m.str());
  }

  //checks data specifications
  if (data.extent(1) != 3 || (size_t)data.extent(2) != m_height || 
      (size_t)data.extent(3) != m_width) {
    boost::format m("input data extents for each frame (the last 3 dimensions of your 4D input array = %dx%dx%d) do not conform to expected format (3x%dx%d), while writing data to file `%s'");
    m % data.extent(1) % data.extent(2) % data.extent(3)
      % m_height % m_width % m_filename;
    throw std::runtime_error(m.str());
  }

  blitz::Range a = blitz::Range::all();
  for(int i=data.lbound(0); i<(data.extent(0)+data.lbound(0)); ++i) {
    ffmpeg::write_video_frame(data(i, a, a, a), m_filename, m_format_context,
        m_stream, m_context_frame, m_rgb24_frame, m_swscaler, m_buffer,
        FFMPEG_VIDEO_BUFFER_SIZE);
    ++m_current_frame;
    m_typeinfo_video.shape[0] += 1;
  }
}

void bob::io::VideoWriter::append(const blitz::Array<uint8_t,3>& data) {
  if (!m_opened) {
    boost::format m("video writer for file `%s' is closed and cannot be written to");
    m % m_filename;
    throw std::runtime_error(m.str());
  }

  //checks data specifications
  if (data.extent(0) != 3 || (size_t)data.extent(1) != m_height || 
      (size_t)data.extent(2) != m_width) {
    boost::format m("input data extents (%dx%dx%d) do not conform to expected format (3x%dx%d), while writing data to file `%s'");
    m % data.extent(0) % data.extent(1) % data.extent(2)
      % m_height % m_width % m_filename;
    throw std::runtime_error(m.str());
  }

  ffmpeg::write_video_frame(data, m_filename, m_format_context,
      m_stream, m_context_frame, m_rgb24_frame, m_swscaler, m_buffer,
      FFMPEG_VIDEO_BUFFER_SIZE);
  ++m_current_frame;
  m_typeinfo_video.shape[0] += 1;
}

void bob::io::VideoWriter::append(const bob::core::array::interface& data) {
  if (!m_opened) {
    boost::format m("video writer for file `%s' is closed and cannot be written to");
    m % m_filename;
    throw std::runtime_error(m.str());
  }

  const bob::core::array::typeinfo& type = data.type();

  if ( type.dtype != bob::core::array::t_uint8 ) {
    boost::format m("input data type = `%s' does not conform to the specified input specifications (3D array = `%s' or 4D array = `%s'), while writing data to file `%s'");
    m % type.str() % m_typeinfo_frame.str() % m_typeinfo_video.str()
      % m_filename;
  }

  if ( type.nd == 3 ) { //appends single frame
    if ( (type.shape[0] != 3) || 
        (type.shape[1] != m_height) || 
        (type.shape[2] != m_width) ) {
      boost::format m("input data extents (%dx%dx%d) do not conform to expected format (3x%dx%d), while writing data to file `%s'");
      m % type.shape[0] % type.shape[1] % type.shape[2]
        % m_height % m_width % m_filename;
      throw std::runtime_error(m.str());
    }

    blitz::TinyVector<int,3> shape;
    shape = 3, m_height, m_width;
    blitz::Array<uint8_t,3> tmp(const_cast<uint8_t*>(static_cast<const uint8_t*>(data.ptr())), shape,
        blitz::neverDeleteData);
    ffmpeg::write_video_frame(tmp, m_filename, m_format_context,
        m_stream, m_context_frame, m_rgb24_frame, m_swscaler, m_buffer,
        FFMPEG_VIDEO_BUFFER_SIZE);
    ++m_current_frame;
    m_typeinfo_video.shape[0] += 1;
  }
  
  else if ( type.nd == 4 ) { //appends a sequence of frames
    if ( (type.shape[1] != 3) || 
         (type.shape[2] != m_height) || 
         (type.shape[3] != m_width) ) {
      boost::format m("input data extents for each frame (the last 3 dimensions of your 4D input array = %dx%dx%d) do not conform to expected format (3x%dx%d), while writing data to file `%s'");
      m % type.shape[1] % type.shape[2] % type.shape[3]
        % m_height % m_width % m_filename;
      throw std::runtime_error(m.str());
    }
    
    blitz::TinyVector<int,3> shape;
    shape = 3, m_height, m_width;
    unsigned long int frame_size = 3 * m_height * m_width;
    uint8_t* ptr = const_cast<uint8_t*>(static_cast<const uint8_t*>(data.ptr()));

    for(size_t i=0; i<type.shape[0]; ++i) {
      blitz::Array<uint8_t,3> tmp(ptr, shape, blitz::neverDeleteData);
      ffmpeg::write_video_frame(tmp, m_filename, m_format_context,
          m_stream, m_context_frame, m_rgb24_frame, m_swscaler, m_buffer,
          FFMPEG_VIDEO_BUFFER_SIZE);
      ++m_current_frame;
      m_typeinfo_video.shape[0] += 1;
      ptr += frame_size;
    }
  }

  else {
    boost::format m("input data type information = `%s' does not conform to the specified input specifications (3D array = `%s' or 4D array = `%s'), while writing data to file `%s'");
    m % type.str() % m_typeinfo_frame.str() % m_typeinfo_video.str()
      % m_filename;
  }

}
