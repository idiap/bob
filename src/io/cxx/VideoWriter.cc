/**
 * @file io/cxx/VideoWriter.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Implementation of a video writer based on ffmpeg, from example output
 * program:
 * http://cekirdek.pardus.org.tr/~ismail/ffmpeg-docs/output-example_8c-source.html.
 * FFMpeg versions for your reference
 * ffmpeg | avformat | avcodec  | avutil   | swscale | old style | swscale GPL?
 * =======+==========+==========+==========+=========+===========+=============
 * 0.5    | 52.31.0  | 52.20.0  | 49.15.0  | 0.7.1   | yes       | yes
 * 0.5.1  | 52.31.0  | 52.20.1  | 49.15.0  | 0.7.1   | yes       | yes
 * 0.5.2  | 52.31.0  | 52.20.1  | 49.15.0  | 0.7.1   | yes       | yes
 * 0.5.3  | 52.31.0  | 52.20.1  | 49.15.0  | 0.7.1   | yes       | yes
 * 0.6    | 52.64.2  | 52.72.2  | 50.15.1  | 0.11.0  | no        | no
 * 0.6.1  | 52.64.2  | 52.72.2  | 50.15.1  | 0.11.0  | no        | no
 * 0.7    | 52.110.0 | 52.122.0 | 50.43.0  | 0.14.1  | no        | no
 * 0.7.1  | 52.110.0 | 52.122.0 | 50.43.0  | 0.14.1  | no        | no
 * 0.8    | 53.4.0   | 53.7.0   | 51.9.1   | 2.0.0   | no        | no
 * 0.11.1 | 54.6.100 | 54.23.100| 51.54.100| 2.1.100 | no        | no
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

#include "bob/io/VideoWriter.h"

#include <boost/format.hpp>
#include <boost/preprocessor.hpp>

#include "bob/io/VideoException.h"

extern "C" {
#include <libavutil/mathematics.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

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

namespace io = bob::io;
namespace ca = bob::core::array;

io::VideoWriter::VideoWriter(const std::string& filename, size_t height,
    size_t width, float framerate, float bitrate, size_t gop):
  m_filename(filename),
  m_isopen(true),
  m_height(height),
  m_width(width),
  m_framerate(framerate),
  m_bitrate(bitrate),
  m_gop(gop),
  m_codecname(""),
  m_codecname_long(""),

  m_oformat_ctxt(0),
  m_format_ctxt(0),
  m_video_stream(0),
  picture(0),
  tmp_picture(0),
  video_outbuf(0),
  video_outbuf_size(0),

  m_current_frame(0),
  m_sws_context(0)
{
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

  //opens the video file and wait for user input

  if (m_height%2 != 0 || m_height == 0 || m_width%2 != 0 || m_width == 0) {
    throw io::FFmpegException(m_filename.c_str(), "height and width must be non-zero multiples of 2");
  }

  // auto detects the output format from the name. default is mpeg.
  // ffmpeg 0.6 and above [libavformat 52.64.2 = 0x344002]
#if LIBAVFORMAT_VERSION_INT >= 0x344002
  m_oformat_ctxt = av_guess_format(NULL, filename.c_str(), NULL);
#else
  m_oformat_ctxt = guess_format(NULL, filename.c_str(), NULL);
#endif

  // by default use the "mpeg" encoder
  if (!m_oformat_ctxt) {
#if LIBAVFORMAT_VERSION_INT >= 0x344002
    m_oformat_ctxt = av_guess_format("mpeg", NULL, NULL);
#else
    m_oformat_ctxt = guess_format("mpeg", NULL, NULL);
#endif
  }

  if (!m_oformat_ctxt) {
    throw io::FFmpegException(m_filename.c_str(), "could not find suitable output format");
  }

  // allocates the output media context
  m_format_ctxt = avformat_alloc_context();
  if (!m_format_ctxt) {
    throw io::FFmpegException(m_filename.c_str(), "could not allocate the output format context");
  }
  m_format_ctxt->oformat = m_oformat_ctxt;

  // if I could not get a codec ID, raise an exception
  if (m_oformat_ctxt->video_codec == CODEC_ID_NONE)
    throw io::FFmpegException(m_filename.c_str(), "could not find suitable codec for encoding");

  // sets the codecnames (short and long versions)
  AVCodec* codec = avcodec_find_decoder(m_oformat_ctxt->video_codec);
  m_codecname = codec->name;
  m_codecname_long = codec->long_name;

  // adds the video stream using the default format codec and initializes it
  m_video_stream = add_video_stream();

  // Sets parameters of output video file
  // ffmpeg 0.7 and above [libavformat 52.110.0 = 0x346e00] doesn't require it
# if LIBAVFORMAT_VERSION_INT < 0x346e00
  // sets the output parameters (must be done even if no parameters).
  if (av_set_parameters(m_format_ctxt, NULL) < 0) {
    throw io::FFmpegException(m_filename.c_str(), "invalid output parameters");
  }

  dump_format(m_format_ctxt, 0, m_filename.c_str(), 1);
# endif

  // now that all the parameters are set, we can open the video codecs
  // and allocate the necessary encode buffers
  open_video();

  // opens the output file, if needed
  if (!(m_oformat_ctxt->flags & AVFMT_NOFILE)) {
#   if LIBAVFORMAT_VERSION_INT >= 0x346e00 && LIBAVFORMAT_VERSION_INT < 0x350400
    if (avio_open(&m_format_ctxt->pb, m_filename.c_str(), URL_WRONLY) < 0) 
#   elif LIBAVFORMAT_VERSION_INT >= 0x350400
    if (avio_open(&m_format_ctxt->pb, m_filename.c_str(), AVIO_FLAG_WRITE) < 0) 
#   else
    if (url_fopen(&m_format_ctxt->pb, m_filename.c_str(), URL_WRONLY) < 0) 
#   endif
    {
      throw io::FFmpegException(m_filename.c_str(), "cannot open file");
    }
  }

  // writes the stream header, if any
# if LIBAVFORMAT_VERSION_INT >= 0x346e00
  avformat_write_header(m_format_ctxt, NULL);
# else
  av_write_header(m_format_ctxt);
# endif
  
  /**
   * Initializes the software scaler (SWScale) so we can convert images from
   * the movie native format into RGB. You can define which kind of
   * interpolation to perform. Some options from libswscale are:
   * SWS_FAST_BILINEAR, SWS_BILINEAR, SWS_BICUBIC, SWS_X, SWS_POINT, SWS_AREA
   * SWS_BICUBLIN, SWS_GAUSS, SWS_SINC, SWS_LANCZOS, SWS_SPLINE
   */
  m_sws_context = sws_getContext(m_width, m_height, PIX_FMT_RGB24,
      m_width, m_height, m_video_stream->codec->pix_fmt, SWS_BICUBIC, 0, 0, 0);
  if (!m_sws_context) {
    throw io::FFmpegException(m_filename.c_str(), "cannot initialize software scaler");
  }
}

io::VideoWriter::~VideoWriter() {
  if (m_isopen) close();
}

// libavformat >= 54.6.100 && libavcodec >= 54.23.100 == ffmpeg-0.11
#if LIBAVFORMAT_VERSION_INT >= 0x360664 && LIBAVCODEC_VERSION_INT >= 0x361764
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
#endif

static void flush_encoder (const std::string& filename,
    AVFormatContext* format_context, AVStream* stream,
    uint8_t* buffer, size_t buffer_size) {

  const AVCodec* codec = stream->codec->codec;

  //We only need to flush codecs that have delayed data processing
   if (!(codec->capabilities & CODEC_CAP_DELAY)) return;
  
   while (true) {

     // libavformat >= 54.6.100 && libavcodec >= 54.23.100 == ffmpeg-0.11
#if LIBAVFORMAT_VERSION_INT >= 0x360664 && LIBAVCODEC_VERSION_INT >= 0x361764

     /* encode the image */
     boost::shared_ptr<AVPacket> pkt = make_packet();

     int got_output;
     int ok = avcodec_encode_video2(stream->codec, pkt.get(), 0, &got_output);

     if (ok < 0) {
       boost::format m("ffmpeg::avcodec_encode_video2() failed: failed to encode video frame while writing to file `%s' -- ffmpeg reports error %d == `%s'");
       m % filename % ok % ffmpeg_error(ok);
       throw std::runtime_error(m.str());
     }

     /* If size is zero, it means the image was buffered. */
     else if (got_output) {
       if (stream->codec->coded_frame->key_frame) pkt->flags |= AV_PKT_FLAG_KEY;
       pkt->stream_index = stream->index;

       /* Write the compressed frame to the media file. */
       ok = av_interleaved_write_frame(format_context, pkt.get());
       if (ok && (ok != AVERROR(EINVAL))) {
         boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to encode video frame while flushing remaining frames to file `%s' -- ffmpeg reports error %d == `%s'");
         m % filename % ok % ffmpeg_error(ok);
         throw std::runtime_error(m.str());
       }
     }

     /* encoded the video, but no pkt got back => video is flushed */
     else if (ok == 0) break;

#else // == #if FFmpeg version < 0.11.0 or using libav

     /**
      * The main difference in this way of encoding is what we are
      * responsible for allocating ourselves the buffer in which the
      * encoding will occur. If that space is too small, encoding the data
      * will result in an error. 
      */

     /* encode the image */
     int out_size = avcodec_encode_video(stream->codec, 
         buffer, buffer_size, 0);

     if (out_size < 0) {
       boost::format m("ffmpeg::avcodec_encode_video() failed: failed to encode video frame while writing to file `%s' -- ffmpeg reports error %d == `%s'");
       m % filename % out_size % ffmpeg_error(out_size);
       throw std::runtime_error(m.str());
     }

     /* If size is zero, it means the image was buffered. */
     else if (out_size > 0) {

       /* encode the image */
       AVPacket pkt;
       av_init_packet(&pkt);

       if ((size_t)stream->codec->coded_frame->pts != AV_NOPTS_VALUE)
         pkt.pts = av_rescale_q(stream->codec->coded_frame->pts,
             stream->codec->time_base, stream->time_base);
       if (stream->codec->coded_frame->key_frame)
         pkt.flags |= AV_PKT_FLAG_KEY;

       pkt.stream_index = stream->index;
       pkt.data         = buffer;
       pkt.size         = out_size;

       /* Write the compressed frame to the media file. */
       int ok = av_interleaved_write_frame(format_context, &pkt);
       if (ok && (ok != AVERROR(EINVAL))) {
         boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to write video frame while encoding file `%s' - ffmpeg reports error %d == `%s'");
         m % filename % ok % ffmpeg_error(ok);
         throw std::runtime_error(m.str());
       }

     }

     /* encoded the video, but no pkt got back => video is flushed */
     else break;

#endif // FFmpeg version >= 0.11.0

   }

}

void io::VideoWriter::close() {

  // flushes the encoder, just in case
  flush_encoder(m_filename, m_format_ctxt, m_video_stream, video_outbuf, video_outbuf_size);

  // writes the trailer, if any.  the trailer must be written
  // before you close the CodecContexts open when you wrote the
  // header; otherwise write_trailer may try to use memory that
  // was freed on av_codec_close()
  if (m_format_ctxt != 0) av_write_trailer(m_format_ctxt);

  // closes each codec opened for the video stream (usually one one)
  if (m_video_stream != 0) close_video();

  // frees the streams
  if (m_format_ctxt != 0) {
    for(int i = 0; i < (int)m_format_ctxt->nb_streams; i++) {
      av_freep(&m_format_ctxt->streams[i]->codec);
      av_freep(&m_format_ctxt->streams[i]);
    }
  }

  // closes the output file
  if (m_oformat_ctxt != 0 && !(m_oformat_ctxt->flags & AVFMT_NOFILE)) 
#   if LIBAVFORMAT_VERSION_INT >= 0x346e00
    avio_close(m_format_ctxt->pb);
#   else
    url_fclose(m_format_ctxt->pb);
#   endif

  // frees the stream
  if (m_format_ctxt != 0) av_free(m_format_ctxt);
  
  // closes the sw scaler context
  if (m_sws_context) { 
    sws_freeContext(m_sws_context);
    m_sws_context=0; 
  }
 
  m_isopen = false;
}

AVStream* io::VideoWriter::add_video_stream() {
  AVCodecContext *c;
  AVStream *st;

# if LIBAVFORMAT_VERSION_INT < 0x350600
  st = av_new_stream(m_format_ctxt, 0);
#else
  st = avformat_new_stream(m_format_ctxt, 0);
#endif
  if (!st) 
    throw io::FFmpegException(m_filename.c_str(), "cannot allocate stream");

  c = st->codec;
  c->codec_id = m_oformat_ctxt->video_codec;
# if LIBAVUTIL_VERSION_INT >= 0x330000
  c->codec_type = AVMEDIA_TYPE_VIDEO;
# else
  c->codec_type = CODEC_TYPE_VIDEO;
# endif

  // puts sample parameters
  c->bit_rate = m_bitrate;
  // resolution must be a multiple of two
  c->width = m_width;
  c->height = m_height;
  // time base: this is the fundamental unit of time (in seconds) in terms
  //      of which frame timestamps are represented. for fixed-fps content,
  //  timebase should be 1/framerate and timestamp increments should be
  //  identically 1.
  c->time_base.den = m_framerate;
  c->time_base.num = 1;
  c->gop_size = m_gop; // emit one intra frame every N frames at most
  c->pix_fmt = PIX_FMT_YUV420P;
  if (c->codec_id == CODEC_ID_MPEG2VIDEO) {
    // just for testing, we also add B frames
    c->max_b_frames = 2;
  }
  if (c->codec_id == CODEC_ID_MPEG1VIDEO) {
    // Needed to avoid using macroblocks in which some coeffs overflow.
    // This does not happen with normal video, it just happens here as
    // the motion of the chroma plane does not match the luma plane.
    c->mb_decision=2;
  }
  // some formats want stream headers to be separate
  if (m_format_ctxt->oformat->flags & AVFMT_GLOBALHEADER)
    c->flags |= CODEC_FLAG_GLOBAL_HEADER;
  return st;
}

AVFrame* io::VideoWriter::alloc_picture(enum PixelFormat pix_fmt) {
  AVFrame* picture = avcodec_alloc_frame();
  
  if (!picture) 
    throw io::FFmpegException(m_filename.c_str(), "cannot allocate frame");

  size_t size = avpicture_get_size(pix_fmt, m_width, m_height);

  uint8_t* picture_buf = (uint8_t*)av_malloc(size);

  if (!picture_buf) {
    av_free(picture);
    throw io::FFmpegException(m_filename.c_str(), "cannot allocate frame buffer");
  }

  avpicture_fill((AVPicture *)picture, picture_buf, pix_fmt, m_width, m_height);
  return picture;
}

void io::VideoWriter::open_video() {
  AVCodec *codec;
  AVCodecContext *c;

  c = m_video_stream->codec;

  // finds the video encoder
  codec = avcodec_find_encoder(c->codec_id);
  if (!codec) throw io::FFmpegException(m_filename.c_str(), "codec not found");

  // opens the codec
# if LIBAVCODEC_VERSION_INT < 0x350700
  if (avcodec_open(c, codec) < 0) {
# else
  if (avcodec_open2(c, codec, 0) < 0) {
# endif
    throw io::FFmpegException(m_filename.c_str(), "cannot open codec");
  }

  video_outbuf = 0;
  if (!(m_format_ctxt->oformat->flags & AVFMT_RAWPICTURE)) {
    // allocate output buffer
    // XXX: API change will be done
    //  buffers passed into lav* can be allocated any way you prefer,
    //  as long as they're aligned enough for the architecture, and
    //  they're freed appropriately (such as using av_free for buffers
    //  allocated with av_malloc)
    video_outbuf_size = 200000;
    video_outbuf = (uint8_t*)av_malloc(video_outbuf_size);

    if (!video_outbuf) 
      throw io::FFmpegException(m_filename.c_str(),
          "cannot allocate output buffer");
  }

  // allocates the encoded raw picture
  picture = alloc_picture(c->pix_fmt);
  picture->pts = 0;

  // if the output format is not RGB24, then a temporary RGB24
  // picture is needed too. It is then converted to the required
  // output format
  if (c->pix_fmt != PIX_FMT_RGB24) tmp_picture = alloc_picture(PIX_FMT_RGB24);
}

/**
 * Transforms from Bob's planar 8-bit RGB representation to whatever is
 * required by the FFmpeg encoder output context (peeked from the AVStream
 * object passed).
 */
static void image_to_context(const blitz::Array<uint8_t,3>& data,
    AVStream* stream, SwsContext* scaler, 
    AVFrame* output_frame, AVFrame* tmp_frame) {

  int width = stream->codec->width;
  int height = stream->codec->height;

  // replace data in the buffer frame by the pixmap to encode
  tmp_frame->linesize[0] = width*3;
  blitz::Array<uint8_t,3> ordered(tmp_frame->data[0],
      blitz::shape(height, width, 3), blitz::neverDeleteData);
  ordered =
    const_cast<blitz::Array<uint8_t,3>&>(data).transpose(1,2,0);
  //organize for ffmpeg

  int ok = sws_scale(scaler, tmp_frame->data, tmp_frame->linesize,
      0, height, output_frame->data, output_frame->linesize);
  if (ok < 0) {
    boost::format m("ffmpeg::sws_scale() failed: could not scale frame while encoding - ffmpeg reports error %d");
    m % ok;
    throw std::runtime_error(m.str());
  }
}

static void write_video_frame (const blitz::Array<uint8_t,3>& data,
    const std::string& filename, AVFormatContext* format_context,
    AVStream* stream, AVFrame* context_frame, AVFrame* tmp_frame,
    SwsContext* swscaler, uint8_t* buffer, size_t buffer_size) {

  image_to_context(data, stream, swscaler, context_frame, tmp_frame);

  if (format_context->oformat->flags & AVFMT_RAWPICTURE) {

    /* Raw video case - directly store the picture in the packet */
    AVPacket pkt;
    av_init_packet(&pkt);

    pkt.flags        |= AV_PKT_FLAG_KEY;
    pkt.stream_index  = stream->index;
    pkt.data          = context_frame->data[0];
    pkt.size          = sizeof(AVPicture);

    int ok = av_interleaved_write_frame(format_context, &pkt);
    if (ok && (ok != AVERROR(EINVAL))) {
      boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to write video frame while encoding file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }

  }

  // libavformat >= 54.6.100 && libavcodec >= 54.23.100 == ffmpeg-0.11
#if LIBAVFORMAT_VERSION_INT >= 0x360664 && LIBAVCODEC_VERSION_INT >= 0x361764

  else {

    /* encode the image */
    boost::shared_ptr<AVPacket> pkt = make_packet();

    int got_output;
    int ok = avcodec_encode_video2(stream->codec, pkt.get(), context_frame, &got_output);
    if (ok < 0) {
      boost::format m("ffmpeg::avcodec_encode_video2() failed: failed to encode video frame while writing to file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }

    /* If size is zero, it means the image was buffered. */
    if (got_output) {
      if (stream->codec->coded_frame->key_frame) pkt->flags |= AV_PKT_FLAG_KEY;
      pkt->stream_index = stream->index;

      /* Write the compressed frame to the media file. */
      ok = av_interleaved_write_frame(format_context, pkt.get());
      if (ok && (ok != AVERROR(EINVAL))) {
        boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to write video frame while encoding file `%s' - ffmpeg reports error %d == `%s'");
        m % filename % ok % ffmpeg_error(ok);
        throw std::runtime_error(m.str());
      }

    }

  }

#else // == #if FFmpeg version < 0.11.0 or using libav

  /**
   * The main difference in this way of encoding is what we are responsible
   * for allocating ourselves the buffer in which the encoding will occur. If
   * that space is too small, encoding the data will result in an error. 
   */

  else {
    /* encode the image */
    int out_size = avcodec_encode_video(stream->codec, 
        buffer, buffer_size, context_frame);

    if (out_size < 0) {
      boost::format m("ffmpeg::avcodec_encode_video() failed: failed to encode video frame while writing to file `%s' -- ffmpeg reports error %d == `%s'");
      m % filename % out_size % ffmpeg_error(out_size);
      throw std::runtime_error(m.str());
    }

    /* If size is zero, it means the image was buffered. */
    if (out_size > 0) {

      /* encode the image */
      AVPacket pkt;
      av_init_packet(&pkt);

      if ((size_t)stream->codec->coded_frame->pts != AV_NOPTS_VALUE)
        pkt.pts = av_rescale_q(stream->codec->coded_frame->pts,
            stream->codec->time_base, stream->time_base);
      if (stream->codec->coded_frame->key_frame)
        pkt.flags |= AV_PKT_FLAG_KEY;

      pkt.stream_index = stream->index;
      pkt.data         = buffer;
      pkt.size         = out_size;

      /* Write the compressed frame to the media file. */
      int ok = av_interleaved_write_frame(format_context, &pkt);
      av_free_packet(&pkt);
      if (ok && (ok != AVERROR(EINVAL))) {
        boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to write video frame while encoding file `%s' - ffmpeg reports error %d == `%s'");
        m % filename % ok % ffmpeg_error(ok);
        throw std::runtime_error(m.str());
      }

    }

  }

#endif // FFmpeg version >= 0.11.0

  context_frame->pts += 1;
}

void io::VideoWriter::close_video() {
  avcodec_close(m_video_stream->codec);
  av_free(picture->data[0]);
  av_free(picture);
  if (tmp_picture) {
    av_free(tmp_picture->data[0]);
    av_free(tmp_picture);
  }
  av_free(video_outbuf);
}

void io::VideoWriter::append(const blitz::Array<uint8_t,3>& data) {
  //are we still opened?
  if (!m_isopen) throw io::VideoIsClosed(m_filename.c_str());

  //checks data specifications
  if (data.extent(0) != 3 || (size_t)data.extent(1) != m_height || 
      (size_t)data.extent(2) != m_width) {
    throw io::FFmpegException(m_filename.c_str(), 
        "input data does not conform with video specifications");
  }
  write_video_frame(data, m_filename, m_format_ctxt,
      m_video_stream, picture, tmp_picture, m_sws_context,
      video_outbuf, video_outbuf_size);
  ++m_current_frame;
  m_typeinfo_video.shape[0] += 1;
}

void io::VideoWriter::append(const blitz::Array<uint8_t,4>& data) {
  //are we still opened?
  if (!m_isopen) throw io::VideoIsClosed(m_filename.c_str());

  //checks data specifications
  if (data.extent(1) != 3 || (size_t)data.extent(2) != m_height || 
      (size_t)data.extent(3) != m_width) {
    throw io::FFmpegException(m_filename.c_str(), 
        "input data does not conform with video specifications");
  }

  blitz::Range a = blitz::Range::all();
  for(int i=data.lbound(0); i<(data.extent(0)+data.lbound(0)); ++i) {
    write_video_frame(data(i, a, a, a), m_filename, m_format_ctxt,
        m_video_stream, picture, tmp_picture, m_sws_context,
        video_outbuf, video_outbuf_size);
    ++m_current_frame;
    m_typeinfo_video.shape[0] += 1;
  }
}

void io::VideoWriter::append(const ca::interface& data) {
  //are we still opened?
  if (!m_isopen) throw io::VideoIsClosed(m_filename.c_str());

  const ca::typeinfo& type = data.type();

  if ( type.dtype != bob::core::array::t_uint8 ) {
    throw io::FFmpegException(m_filename.c_str(), 
        "input data type does not conform with video specifications");
  }

  if ( type.nd == 3 ) { //appends single frame
    if ( (type.shape[0] != 3) || 
         (type.shape[1] != m_height) || 
         (type.shape[2] != m_width) ) {
      throw io::FFmpegException(m_filename.c_str(), 
          "input data shape does not conform with video specifications");
    }
    
    blitz::TinyVector<int,3> shape;
    shape = 3, m_height, m_width;
    blitz::Array<uint8_t,3> tmp(const_cast<uint8_t*>(static_cast<const uint8_t*>(data.ptr())), shape, blitz::neverDeleteData);
    write_video_frame(tmp, m_filename, m_format_ctxt,
        m_video_stream, picture, tmp_picture, m_sws_context,
        video_outbuf, video_outbuf_size);
    ++m_current_frame;
    m_typeinfo_video.shape[0] += 1;
  }
  
  else if ( type.nd == 4 ) { //appends a sequence of frames
    if ( (type.shape[1] != 3) || 
         (type.shape[2] != m_height) || 
         (type.shape[3] != m_width) ) {
      throw io::FFmpegException(m_filename.c_str(), 
          "input data shape does not conform with video specifications");
    }
    
    blitz::TinyVector<int,3> shape;
    shape = 3, m_height, m_width;
    unsigned long int frame_size = 3 * m_height * m_width;
    uint8_t* ptr = const_cast<uint8_t*>(static_cast<const uint8_t*>(data.ptr()));

    for(size_t i=0; i<type.shape[0]; ++i) {
      blitz::Array<uint8_t,3> tmp(ptr, shape, blitz::neverDeleteData);
      write_video_frame(tmp, m_filename, m_format_ctxt,
          m_video_stream, picture, tmp_picture, m_sws_context,
          video_outbuf, video_outbuf_size);
      ++m_current_frame;
      m_typeinfo_video.shape[0] += 1;
      ptr += frame_size;
    }
  }

  else {
    throw io::FFmpegException(m_filename.c_str(), 
        "input data dimensions do not conform with video specifications");
  }

}

std::string io::VideoWriter::info() const {
  /**
   * This will create a local description of the contents of the stream, in
   * printable format.
   */
  boost::format info("Video file: %s; FFmpeg: avformat-%s; avcodec-%s; avutil-%s; swscale-%d; Codec: %s (%s); Time: %.2f s (%d @ %2.fHz); Size (w x h): %d x %d pixels");
  info % m_filename;
  info % BOOST_PP_STRINGIZE(LIBAVFORMAT_VERSION);
  info % BOOST_PP_STRINGIZE(LIBAVCODEC_VERSION);
  info % BOOST_PP_STRINGIZE(LIBAVUTIL_VERSION);
  info % BOOST_PP_STRINGIZE(LIBSWSCALE_VERSION);
  info % m_codecname_long;
  info % m_codecname;
  info % (m_current_frame/m_framerate);
  info % m_current_frame;
  info % m_framerate;
  info % m_width;
  info % m_height;
  return info.str();
}
