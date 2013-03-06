/**
 * @file io/cxx/VideoReader.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Implements a class to read and write Video files and convert the frames into
 * something that bob can understand. This implementation is heavily based on
 * the excellent tutorial here: http://dranger.com/ffmpeg/, with some personal
 * modifications.
 * FFMpeg versions for your reference
 * ffmpeg | avformat | avcodec  | avutil  | swscale | old style | swscale GPL?
 * =======+==========+==========+=========+=========+===========+==============
 * 0.5    | 52.31.0  | 52.20.0  | 49.15.0 | 0.7.1   | yes       | yes
 * 0.5.1  | 52.31.0  | 52.20.1  | 49.15.0 | 0.7.1   | yes       | yes
 * 0.5.2  | 52.31.0  | 52.20.1  | 49.15.0 | 0.7.1   | yes       | yes
 * 0.5.3  | 52.31.0  | 52.20.1  | 49.15.0 | 0.7.1   | yes       | yes
 * 0.6    | 52.64.2  | 52.72.2  | 50.15.1 | 0.11.0  | no        | no
 * 0.6.1  | 52.64.2  | 52.72.2  | 50.15.1 | 0.11.0  | no        | no
 * 0.7    | 52.110.0 | 52.122.0 | 50.43.0 | 0.14.1  | no        | no
 * 0.7.1  | 52.110.0 | 52.122.0 | 50.43.0 | 0.14.1  | no        | no
 * 0.8    | 53.4.0   | 53.7.0   | 51.9.1  | 2.0.0   | no        | no
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

#include "bob/io/VideoReader.h"

#include <stdexcept>
#include <boost/format.hpp>
#include <boost/preprocessor.hpp>
#include <limits>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include "bob/core/array_check.h"
#include "bob/io/Exception.h"
#include "bob/io/VideoException.h"
#include "bob/core/blitz_array.h"
#include "bob/core/logging.h"

static std::string ffmpeg_error(int num) {
#if LIBAVUTIL_VERSION_INT >= 0x320f01 //50.15.1 @ ffmpeg-0.6
  static const int ERROR_SIZE = 1024;
  char message[ERROR_SIZE];
  int ok = av_strerror(num, message, ERROR_SIZE);
  if (ok < 0) {
    throw std::runtime_error("ffmpeg::av_strerror() failed to report - maybe you have a memory issue?");
  }
  return std::string(message);
#else
  return std::string("unknown error - ffmpeg version < 0.6");
#endif
}

static AVPacket* allocate_packet() {
  AVPacket* retval = new AVPacket;
  av_init_packet(retval);
  retval->data = 0;
  retval->size = 0;
  return retval;
}

static void deallocate_packet(AVPacket* p) {
    if (p->size || p->data) av_free_packet(p);
      delete p;
}

static boost::shared_ptr<AVPacket> make_packet() {
    return boost::shared_ptr<AVPacket>(allocate_packet(), 
              std::ptr_fun(deallocate_packet));
}

namespace io = bob::io;
namespace ca = bob::core::array;

io::VideoReader::VideoReader(const std::string& filename):
  m_filepath(filename),
  m_height(0),
  m_width(0),
  m_nframes(0),
  m_framerate(0),
  m_duration(0),
  m_codecname(""),
  m_codecname_long(""),
  m_formatted_info("")
{
  open();
}

io::VideoReader::VideoReader(const io::VideoReader& other):
  m_filepath(other.m_filepath),
  m_height(0),
  m_width(0),
  m_nframes(0),
  m_framerate(0),
  m_duration(0),
  m_codecname(""),
  m_codecname_long(""),
  m_formatted_info("")
{
  open();
}

io::VideoReader& io::VideoReader::operator= (const io::VideoReader& other) {
  m_filepath = other.m_filepath;
  m_height = 0;
  m_width = 0;
  m_nframes = 0;
  m_framerate = 0.0;
  m_duration = 0;
  m_codecname = "";
  m_codecname_long = "";
  m_formatted_info = "";
  open();
  return *this;
}

void io::VideoReader::open() {
  AVFormatContext* format_ctxt = 0;

  // Opens a video file
  // ffmpeg 0.7 and above [libavformat 52.122.0 = 0x347a00]
# if LIBAVCODEC_VERSION_INT >= 0x347a00
  if (avformat_open_input(&format_ctxt, m_filepath.c_str(), NULL, NULL) != 0) 
# else
  if (av_open_input_file(&format_ctxt, m_filepath.c_str(), NULL, 0, NULL) != 0) 
# endif
  {
    throw io::FileNotReadable(m_filepath);
  }

  // Retrieve stream information
# if LIBAVFORMAT_VERSION_INT < 0x350600
  if (av_find_stream_info(format_ctxt)<0) {
    av_close_input_file(format_ctxt);
# else
  if (avformat_find_stream_info(format_ctxt, 0)<0) {
    avformat_close_input(&format_ctxt);
# endif
    throw io::FFmpegException(m_filepath.c_str(), "cannot find stream info");
  }

  // Look for the first video stream in the file
  int stream_index = -1;
  for (size_t i=0; i<format_ctxt->nb_streams; ++i) {
#   if LIBAVUTIL_VERSION_INT >= 0x330000
    if (format_ctxt->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO) 
#   else
    if (format_ctxt->streams[i]->codec->codec_type==CODEC_TYPE_VIDEO) 
#   endif
    {
      stream_index = i;
      break;
    }
  }
  if(stream_index == -1) {
# if LIBAVFORMAT_VERSION_INT < 0x350600
    av_close_input_file(format_ctxt);
# else
    avformat_close_input(&format_ctxt);
# endif
    throw io::FFmpegException(m_filepath.c_str(), "cannot find any video stream");
  }

  // Get a pointer to the codec context for the video stream
  AVCodecContext* codec_ctxt = format_ctxt->streams[stream_index]->codec;

  // Hack to correct frame rates that seem to be generated by some codecs 
  if(codec_ctxt->time_base.num > 1000 && codec_ctxt->time_base.den == 1) {
    codec_ctxt->time_base.den = 1000;
  }

  // Find the decoder for the video stream
  AVCodec* codec = avcodec_find_decoder(codec_ctxt->codec_id);

  if (!codec) {
# if LIBAVFORMAT_VERSION_INT < 0x350600
    av_close_input_file(format_ctxt);
# else
    avformat_close_input(&format_ctxt);
# endif
    throw io::FFmpegException(m_filepath.c_str(), "unsupported codec required");
  }

  // Open codec
# if LIBAVCODEC_VERSION_INT < 0x350700
  if (avcodec_open(codec_ctxt, codec) < 0) {
# else
  if (avcodec_open2(codec_ctxt, codec, 0) < 0) {
# endif
# if LIBAVFORMAT_VERSION_INT < 0x350600
    av_close_input_file(format_ctxt);
# else
    avformat_close_input(&format_ctxt);
# endif
    throw io::FFmpegException(m_filepath.c_str(), "cannot open supported codec");
  }

  /**
   * Copies some information from the contexts opened
   */
  m_width = codec_ctxt->width;
  m_height = codec_ctxt->height;
  m_duration = format_ctxt->duration;
  m_nframes = format_ctxt->streams[stream_index]->nb_frames;
  if (m_nframes > 0) {
    //number of frames is known
    m_framerate = m_nframes * AV_TIME_BASE / m_duration;
  }
  else {
    //number of frames is not known
    m_framerate = av_q2d(format_ctxt->streams[stream_index]->r_frame_rate);
    m_nframes = (int)(m_framerate * m_duration / AV_TIME_BASE);
  }
  m_codecname = codec->name;
  m_codecname_long = codec->long_name;

  /**
   * This will create a local description of the contents of the stream, in
   * printable format.
   */
  boost::format fmt("Video file: %s; FFmpeg: avformat-%s; avcodec-%s; avutil-%s; swscale-%d; Codec: %s (%s); Time: %.2f s (%d @ %2.fHz); Size (w x h): %d x %d pixels");
  fmt % m_filepath;
  fmt % BOOST_PP_STRINGIZE(LIBAVFORMAT_VERSION);
  fmt % BOOST_PP_STRINGIZE(LIBAVCODEC_VERSION);
  fmt % BOOST_PP_STRINGIZE(LIBAVUTIL_VERSION);
  fmt % BOOST_PP_STRINGIZE(LIBSWSCALE_VERSION);
  fmt % m_codecname_long;
  fmt % m_codecname;
  fmt % (m_duration / 1e6);
  fmt % m_nframes;
  fmt % m_framerate;
  fmt % m_width;
  fmt % m_height;
  m_formatted_info = fmt.str();

  /**
   * This will make sure we can interface with the io subsystem
   */
  m_typeinfo_video.dtype = m_typeinfo_frame.dtype = core::array::t_uint8;
  m_typeinfo_video.nd = 4;
  m_typeinfo_frame.nd = 3;
  m_typeinfo_video.shape[0] = m_nframes;
  m_typeinfo_video.shape[1] = m_typeinfo_frame.shape[0] = 3;
  m_typeinfo_video.shape[2] = m_typeinfo_frame.shape[1] = m_height;
  m_typeinfo_video.shape[3] = m_typeinfo_frame.shape[2] = m_width;
  m_typeinfo_frame.update_strides();
  m_typeinfo_video.update_strides();

  //closes the codec we used
  avcodec_close(codec_ctxt);

  //and we close the input file
# if LIBAVFORMAT_VERSION_INT < 0x350600
  av_close_input_file(format_ctxt);
# else
  avformat_close_input(&format_ctxt);
# endif
}

io::VideoReader::~VideoReader() {
}

size_t io::VideoReader::load(blitz::Array<uint8_t,4>& data,
  bool throw_on_error) const {
  ca::blitz_array tmp(data);
  return load(tmp, throw_on_error);
}

size_t io::VideoReader::load(ca::interface& b, 
  bool throw_on_error) const {

  //checks if the output array shape conforms to the video specifications,
  //otherwise, throw.
  if (!m_typeinfo_video.is_compatible(b.type())) {
    boost::format s("input buffer (%s) does not conform to the video size specifications (%s)");
    s % b.type().str() % m_typeinfo_video.str();
    throw std::invalid_argument(s.str());
  }

  unsigned long int frame_size = m_typeinfo_frame.buffer_size();
  uint8_t* ptr = static_cast<uint8_t*>(b.ptr());
  size_t frames_read = 0;

  for (const_iterator it=begin(); it!=end();) {
    ca::blitz_array ref(static_cast<void*>(ptr), m_typeinfo_frame);
    if (it.read(ref, throw_on_error)) {
      ptr += frame_size;
      ++frames_read;
    }
    //otherwise we don't count!
  }

  return frames_read;
}

io::VideoReader::const_iterator io::VideoReader::begin() const {
  return io::VideoReader::const_iterator(this);
}

io::VideoReader::const_iterator io::VideoReader::end() const {
  return io::VideoReader::const_iterator();
}

/**
 * iterator implementation
 */

io::VideoReader::const_iterator::const_iterator(const io::VideoReader* parent) :
  m_parent(parent),
  m_format_ctxt(0),
  m_stream_index(-1),
  m_codec_ctxt(0),
  m_codec(0),
  m_frame_buffer(0),
  m_rgb_frame_buffer(0),
  m_raw_buffer(0),
  m_current_frame(std::numeric_limits<size_t>::max()),
  m_sws_context(0)
{
  init();
}

io::VideoReader::const_iterator::const_iterator():
  m_parent(0),
  m_format_ctxt(0),
  m_stream_index(-1),
  m_codec_ctxt(0),
  m_codec(0),
  m_frame_buffer(0),
  m_rgb_frame_buffer(0),
  m_raw_buffer(0),
  m_current_frame(std::numeric_limits<size_t>::max()),
  m_sws_context(0)
{
}

io::VideoReader::const_iterator::const_iterator
(const io::VideoReader::const_iterator& other) :
  m_parent(other.m_parent),
  m_format_ctxt(0),
  m_stream_index(-1),
  m_codec_ctxt(0),
  m_codec(0),
  m_frame_buffer(0),
  m_rgb_frame_buffer(0),
  m_raw_buffer(0),
  m_current_frame(std::numeric_limits<size_t>::max()),
  m_sws_context(0)
{
  init();
  (*this) += other.m_current_frame;
}

io::VideoReader::const_iterator::~const_iterator() {
  reset();
}

io::VideoReader::const_iterator& io::VideoReader::const_iterator::operator= (const io::VideoReader::const_iterator& other) {
  reset();
  m_parent = other.m_parent;
  init();
  (*this) += other.m_current_frame;
  return *this;
}

void io::VideoReader::const_iterator::init() {
  const char* filename = m_parent->filename().c_str();

  //basic constructor, prepare readout
  // Opens a video file
  // ffmpeg 0.7 and above [libavformat 53.0.0 = 0x350000]
# if LIBAVCODEC_VERSION_INT >= 0x347a00
  if (avformat_open_input(&m_format_ctxt, filename, NULL, NULL) != 0) 
# else
  if (av_open_input_file(&m_format_ctxt, filename, NULL, 0, NULL) != 0) 
# endif
  {
    throw io::FileNotReadable(filename);
  }

  // Retrieve stream information
# if LIBAVFORMAT_VERSION_INT < 0x350000
  if (av_find_stream_info(m_format_ctxt)<0) {
# else
  if (avformat_find_stream_info(m_format_ctxt, 0)<0) {
# endif
    throw io::FFmpegException(filename, "cannot find stream info");
  }

  // Look for the first video stream in the file
  for (size_t i=0; i<m_format_ctxt->nb_streams; ++i) {
#   if LIBAVUTIL_VERSION_INT >= 0x330000
    if (m_format_ctxt->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO) 
#   else
    if (m_format_ctxt->streams[i]->codec->codec_type==CODEC_TYPE_VIDEO) 
#   endif
    {
      m_stream_index = i;
      break;
    }
  }
  if(m_stream_index == -1) {
    throw io::FFmpegException(filename, "cannot find any video stream");
  }

  // Get a pointer to the codec context for the video stream
  m_codec_ctxt = m_format_ctxt->streams[m_stream_index]->codec;

  // Find the decoder for the video stream
  m_codec = avcodec_find_decoder(m_codec_ctxt->codec_id);

  if (!m_codec) {
    throw io::FFmpegException(filename, "unsupported codec required");
  }

  // Open codec
# if LIBAVCODEC_VERSION_INT < 0x350700
  if (avcodec_open(m_codec_ctxt, m_codec) < 0) {
# else
  if (avcodec_open2(m_codec_ctxt, m_codec, 0) < 0) {
# endif
    throw io::FFmpegException(filename, "cannot open supported codec");
  }

  // Hack to correct frame rates that seem to be generated by some codecs 
  if(m_codec_ctxt->time_base.num > 1000 && m_codec_ctxt->time_base.den == 1)
    m_codec_ctxt->time_base.den = 1000;

  // Allocate memory for a buffer to read frames
  m_frame_buffer = avcodec_alloc_frame();
  if (!m_frame_buffer) {
    throw io::FFmpegException(filename, "cannot allocate frame buffer");
  }

  // Allocate memory for a second buffer that contains RGB converted data.
  m_rgb_frame_buffer = avcodec_alloc_frame();
  if (!m_rgb_frame_buffer) {
    throw io::FFmpegException(filename, "cannot allocate RGB frame buffer");
  }

  // Allocate memory for the raw data buffer
  int nbytes = avpicture_get_size(PIX_FMT_RGB24, m_codec_ctxt->width,
      m_codec_ctxt->height);
  m_raw_buffer = (uint8_t*)av_malloc(nbytes*sizeof(uint8_t));
  if (!m_raw_buffer) {
    throw io::FFmpegException(filename, "cannot allocate raw frame buffer");
  }
  
  // Assign appropriate parts of buffer to image planes in m_rgb_frame_buffer
  avpicture_fill((AVPicture *)m_rgb_frame_buffer, m_raw_buffer, PIX_FMT_RGB24,
      m_parent->width(), m_parent->height());

  /**
   * Initializes the software scaler (SWScale) so we can convert images from
   * the movie native format into RGB. You can define which kind of
   * interpolation to perform. Some options from libswscale are:
   * SWS_FAST_BILINEAR, SWS_BILINEAR, SWS_BICUBIC, SWS_X, SWS_POINT, SWS_AREA
   * SWS_BICUBLIN, SWS_GAUSS, SWS_SINC, SWS_LANCZOS, SWS_SPLINE
   */
  m_sws_context = sws_getContext(m_parent->width(), m_parent->height(),
      m_codec_ctxt->pix_fmt, m_parent->width(), m_parent->height(),
      PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
  if (!m_sws_context) {
    throw io::FFmpegException(filename, "cannot initialize software scaler");
  }

  //At this point we are ready to start reading out frames.
  m_current_frame = 0;
  
  //The file maybe valid, but contain zero frames... We check for this here:
  if (m_current_frame >= m_parent->numberOfFrames()) {
    //transform the current iterator in "end"
    reset();
  }
}

void io::VideoReader::const_iterator::reset() {
  //free-up memory
  if (m_frame_buffer) { 
    av_free(m_frame_buffer); 
    m_frame_buffer = 0;
  }
  
  if (m_rgb_frame_buffer) { 
    av_free(m_rgb_frame_buffer); 
    m_rgb_frame_buffer = 0;
  }

  if (m_raw_buffer) { 
    av_free(m_raw_buffer); 
    m_raw_buffer=0; 
  }
  
  if (m_sws_context) { 
    sws_freeContext(m_sws_context);
    m_sws_context=0; 
  }
  
  //closes the codec we used
  if (m_codec_ctxt) {
    avcodec_close(m_codec_ctxt);
    m_codec_ctxt = 0;
    m_codec = 0;
  }
  
  //closes the video file we opened
  if (m_format_ctxt) {
# if LIBAVFORMAT_VERSION_INT < 0x350600
    av_close_input_file(m_format_ctxt);
    m_format_ctxt = 0;
# else
    avformat_close_input(&m_format_ctxt);
# endif
  }

  m_current_frame = std::numeric_limits<size_t>::max(); //that means "end" 

  m_parent = 0;
}

bool io::VideoReader::const_iterator::read(blitz::Array<uint8_t,3>& data,
  bool throw_on_error) {
  ca::blitz_array tmp(data);
  return read(tmp, throw_on_error);
}

static int decode_frame (const std::string& filename, int current_frame,
    AVCodecContext* codec_context, SwsContext* scaler,
    AVFrame* context_frame, uint8_t* data, boost::shared_ptr<AVPacket> pkt,
    int& got_frame, bool throw_on_error) {

  // In this call, 3 things can happen:
  //
  // 1. if ok < 0, an error has been detected
  // 2. if ok >=0, something was read from the file, correctly. In this
  // condition, **only* if "got_frame" == 1, a frame is ready to be decoded.
  //
  // It is **not** an error that ok is >= 0 and got_frame == 0. This, in fact,
  // happens often with recent versions of ffmpeg.

#if LIBAVCODEC_VERSION_INT >= 0x344802 //52.72.2 @ ffmpeg-0.6

  int ok = avcodec_decode_video2(codec_context, context_frame,
      &got_frame, pkt.get());

#else

  int ok = avcodec_decode_video(codec_context, context_frame, &got_frame, pkt->data, pkt->size);

#endif

  if (ok < 0 && throw_on_error) {
    boost::format m("ffmpeg::avcodec_decode_video/2() failed: could not decode frame %d of file `%s' - ffmpeg reports error %d == `%s'");
    m % current_frame % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  if (got_frame) {

    // In this case, we call the software scaler to decode the frame data.
    // Normally, this means converting from planar YUV420 into packed RGB.

    uint8_t* planes[] = {data, 0};
    int linesize[] = {3*codec_context->width, 0};

    int conv_height = sws_scale(scaler, context_frame->data,
        context_frame->linesize, 0, codec_context->height, planes, linesize);

    if (conv_height < 0) {

      if (throw_on_error) {
        boost::format m("ffmpeg::sws_scale() failed: could not scale frame %d of file `%s' - ffmpeg reports error %d");
        m % current_frame % filename % conv_height;
        throw std::runtime_error(m.str());
      }

      return -1;
    }

  }

  return ok;
}

static bool read_video_frame (const std::string& filename, 
    int current_frame, int stream_index, AVFormatContext* format_context,
    AVCodecContext* codec_context, SwsContext* swscaler,
    AVFrame* context_frame, AVFrame* rgb_frame, bool throw_on_error) {

  uint8_t* data = rgb_frame->data[0];

  boost::shared_ptr<AVPacket> pkt = make_packet();

  int ok = av_read_frame(format_context, pkt.get());

  if (ok < 0 && ok != (int)AVERROR_EOF) {
    if (throw_on_error) {
      boost::format m("ffmpeg::av_read_frame() failed: on file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
    else return false;
  }

  int got_frame = 0;

  while(!got_frame) {

    // if we have reached the end-of-file, frames can still be cached
    if (ok == (int)AVERROR_EOF) {
      pkt->data = 0;
      pkt->size = 0;
      decode_frame(filename, current_frame, codec_context, swscaler,
          context_frame, data, pkt, got_frame, throw_on_error);
    }
    else {
      if (pkt->stream_index == stream_index) {
        decode_frame(filename, current_frame, codec_context,
            swscaler, context_frame, data, pkt, got_frame,
            throw_on_error);
      }
    }

  }

  return (got_frame > 0);
}

bool io::VideoReader::const_iterator::read(bob::core::array::interface& data,
    bool throw_on_error) {

  if (!m_parent) {
    //we are already past the end of the stream
    throw std::runtime_error("video iterator for file has already reached its end and was reset");
  }

  //checks if we have not passed the end of the video sequence already
  if(m_current_frame >= m_parent->numberOfFrames()) {

    if (throw_on_error) {
      boost::format m("you are trying to read past the file end (next frame no. to be read would be %d) on file %s, which contains only %d frames");
      m % m_current_frame % m_parent->m_filepath % m_parent->m_nframes;
      throw std::runtime_error(m.str());
    }

    reset();
    return false;
  }

  const bob::core::array::typeinfo& info = data.type();

  //checks if the output array shape conforms to the video specifications,
  //otherwise, throw
  if (!info.is_compatible(m_parent->m_typeinfo_frame)) {
    boost::format s("input buffer (%s) does not conform to the video frame size specifications (%s)");
    s % info.str() % m_parent->m_typeinfo_frame.str();
    throw std::invalid_argument(s.str());
  }

  //we are going to need another copy step - use our internal array
  bool ok = read_video_frame(m_parent->m_filepath, m_current_frame,
      m_stream_index, m_format_ctxt, m_codec_ctxt, m_sws_context,
      m_frame_buffer, m_rgb_frame_buffer, throw_on_error);

  if (ok) {

    //now we copy from one container to the other, using our Blitz++ technique
    blitz::TinyVector<int,3> shape;
    blitz::TinyVector<int,3> stride;

    shape = info.shape[0], info.shape[1], info.shape[2];
    stride = info.stride[0], info.stride[1], info.stride[2];
    blitz::Array<uint8_t,3> dst(static_cast<uint8_t*>(data.ptr()),
        shape, stride, blitz::neverDeleteData);

    shape = info.shape[1], info.shape[2], info.shape[0];
    blitz::Array<uint8_t,3> rgb_array(m_rgb_frame_buffer->data[0], shape,
        blitz::neverDeleteData);
    dst = rgb_array.transpose(2,0,1);
    ++m_current_frame;

  }

  return ok;
}

static int dummy_decode_frame (const std::string& filename, int current_frame,
    AVCodecContext* codec_context, AVFrame* context_frame,
    boost::shared_ptr<AVPacket> pkt, int& got_frame, bool throw_on_error) {

  // In this call, 3 things can happen:
  //
  // 1. if ok < 0, an error has been detected
  // 2. if ok >=0, something was read from the file, correctly. In this
  // condition, **only* if "got_frame" == 1, a frame is ready to be decoded.
  //
  // It is **not** an error that ok is >= 0 and got_frame == 0. This, in fact,
  // happens often with recent versions of ffmpeg.

#if LIBAVCODEC_VERSION_INT >= 0x344802 //52.72.2 @ ffmpeg-0.6

  int ok = avcodec_decode_video2(codec_context, context_frame,
      &got_frame, pkt.get());

#else

  int ok = avcodec_decode_video(codec_context, context_frame, &got_frame, pkt->data, pkt->size);

#endif

  if (ok < 0 && throw_on_error) {
    boost::format m("ffmpeg::avcodec_decode_video/2() failed: could not skip frame %d of file `%s' - ffmpeg reports error %d == `%s'");
    m % current_frame % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  return ok;
}

static bool skip_video_frame (const std::string& filename,
    int current_frame, int stream_index, AVFormatContext* format_context,
    AVCodecContext* codec_context, AVFrame* context_frame,
    bool throw_on_error) {

  boost::shared_ptr<AVPacket> pkt = make_packet();

  int ok = av_read_frame(format_context, pkt.get());

  if (ok < 0 && ok != (int)AVERROR_EOF) {
    if (throw_on_error) {
      boost::format m("ffmpeg::av_read_frame() failed: on file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
    else return false;
  }

  int got_frame = 0;

  // if we have reached the end-of-file, frames can still be cached
  if (ok == (int)AVERROR_EOF) {
    pkt->data = 0;
    pkt->size = 0;
    dummy_decode_frame(filename, current_frame, codec_context,
        context_frame, pkt, got_frame, throw_on_error);
  }
  else {
    if (pkt->stream_index == stream_index) {
      dummy_decode_frame(filename, current_frame, codec_context,
          context_frame, pkt, got_frame, throw_on_error);
    }
  }

  return (got_frame > 0);
}

io::VideoReader::const_iterator& io::VideoReader::const_iterator::operator++ () {
  if (!m_parent) {
    //we are already past the end of the stream
    throw std::runtime_error("video iterator for file has already reached its end and was reset");
  }

  //checks if we have not passed the end of the video sequence already
  if(m_current_frame >= m_parent->numberOfFrames()) {
    reset();
    return *this;
  }

  //we are going to need another copy step - use our internal array
  try {
    bool ok = skip_video_frame(m_parent->m_filepath, m_current_frame,
        m_stream_index, m_format_ctxt, m_codec_ctxt, m_frame_buffer,
        true);
    if (ok) ++m_current_frame;
  }
  catch (std::runtime_error& e) {
    reset();
  }

  return *this;
}

io::VideoReader::const_iterator& io::VideoReader::const_iterator::operator+= (size_t frames) {
  for (size_t i=0; i<frames; ++i) ++(*this);
  return *this;
}

bool io::VideoReader::const_iterator::operator== (const const_iterator& other) {
  return (this->m_parent == other.m_parent) && (this->m_current_frame == other.m_current_frame);
}

bool io::VideoReader::const_iterator::operator!= (const const_iterator& other) {
  return !(*this == other);
}
