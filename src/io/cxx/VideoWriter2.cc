/**
 * @file io/cxx/VideoWriter2.cc
 * @date Wed 28 Nov 2012 13:51:58 CET
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A class to help you write videos. This code originates from
 * http://ffmpeg.org/doxygen/1.0/, "muxing.c" example.
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

#include <boost/format.hpp>

#include "bob/io/VideoWriter2.h"
#include "bob/io/VideoUtilities.h"

#if FFMPEG_VERSION_INT < 0x000800
#error VideoWriter2.cc can only be compiled against FFmpeg >= 0.8.0
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/avstring.h>
#include <libavutil/mathematics.h>
}

#if FFMPEG_VERSION_INT < 0x000b00
#define FFMPEG_VIDEO_BUFFER_SIZE 200000
#endif

/**
 * A copy of avformat_alloc_output_context2() from complete versions of ffmpeg
 * - addresses a lack on Ubuntu 12.04 and 12.10
 */
static int bob_avformat_alloc_output_context2(AVFormatContext **avctx, AVOutputFormat *oformat,
    const char *format, const char *filename)
{
  AVFormatContext *s = avformat_alloc_context();
  int ret = 0;

  *avctx = NULL;
  if (!s)
    goto nomem;

  if (!oformat) {
    if (format) {
      oformat = av_guess_format(format, NULL, NULL);
      if (!oformat) {
        av_log(s, AV_LOG_ERROR, "Requested output format '%s' is not a suitable output format\n", format);
        ret = AVERROR(EINVAL);
        goto error;
      }
    } else {
      oformat = av_guess_format(NULL, filename, NULL);
      if (!oformat) {
        ret = AVERROR(EINVAL);
        av_log(s, AV_LOG_ERROR, "Unable to find a suitable output format for '%s'\n",
            filename);
        goto error;
      }
    }
  }

  s->oformat = oformat;
  if (s->oformat->priv_data_size > 0) {
    s->priv_data = av_mallocz(s->oformat->priv_data_size);
    if (!s->priv_data)
      goto nomem;
    if (s->oformat->priv_class) {
      *(const AVClass**)s->priv_data= s->oformat->priv_class;
      av_opt_set_defaults(s->priv_data);
    }
  } else
    s->priv_data = NULL;

  if (filename)
    av_strlcpy(s->filename, filename, sizeof(s->filename));
  *avctx = s;
  return 0;
nomem:
  av_log(s, AV_LOG_ERROR, "Out of memory\n");
  ret = AVERROR(ENOMEM);
error:
  avformat_free_context(s);
  return ret;
}

/**
 * Safely allocate the output format media context
 */
static AVFormatContext* allocate_format_context(
    const boost::filesystem::path& filename,
    const std::string& formatname) {
  AVFormatContext* retval;

  if (formatname.size() != 0) {
    bob_avformat_alloc_output_context2(&retval, 0, formatname.c_str(), filename.string().c_str());
  }
  else {
    bob_avformat_alloc_output_context2(&retval, 0, 0, filename.string().c_str());
  }

  if (!retval) { //try mpeg encoder
    bob_avformat_alloc_output_context2(&retval, 0, "mpeg", filename.string().c_str());
  }

  if (!retval) {
    boost::format m("ffmpeg::avformat_alloc_output_context2() failed: could not allocate output context based on format name == `%s', filename == `%s' or just specifying a `mpeg' format");
    m % formatname % filename.string();
    throw std::runtime_error(m.str());
  }

  return retval;
}

static void deallocate_format_context(AVFormatContext* f) {
  if (f) {
    av_free(f);
  }
}

static AVCodec* find_encoder(
    const boost::filesystem::path& filename,
    boost::shared_ptr<AVFormatContext> fmtctxt,
    const std::string& codecname
    ) {
  AVCodec* retval = 0;

  /* find the video encoder */
  if (codecname.size() != 0) {
    retval = avcodec_find_encoder_by_name(codecname.c_str());
    if (!retval) {
      boost::format m("ffmpeg::avcodec_find_encoder_by_name(`%s') failed: could not find a suitable codec for encoding video file `%s' using the output format `%s' == `%s'");
      m % codecname % filename.string() % fmtctxt->oformat->name 
        % fmtctxt->oformat->long_name;
      throw std::runtime_error(m.str());
    }
  }
  else {
    if (fmtctxt->oformat->video_codec == AV_CODEC_ID_NONE) {
      boost::format m("could not identify codec for encoding video file `%s'; tried codec with name `%s' first and then tried output format's `%s' == `%s' video_codec entry, which was also null");
      m % filename.string() % fmtctxt->oformat->name
        % fmtctxt->oformat->long_name;
      throw std::runtime_error(m.str());
    }
    retval = avcodec_find_encoder(fmtctxt->oformat->video_codec);

    if (!retval) {
      boost::format m("ffmpeg::avcodec_find_encoder(0x%x) failed: could not find encoder for codec with identifier for encoding video file `%s' using the output format `%s' == `%s'");
      m % fmtctxt->oformat->video_codec % filename.string() 
        % fmtctxt->oformat->name % fmtctxt->oformat->long_name;
      throw std::runtime_error(m.str());
    }
  }

  return retval;
}

static void deallocate_stream(AVStream* s) {
  if (s) {
    av_freep(&s->codec); ///< free the codec context
    av_freep(&s); ///< free the stream itself
  }
}

#if FFMPEG_VERSION_INT >= 0x000900

static AVStream* allocate_stream(
    const boost::filesystem::path& filename,
    boost::shared_ptr<AVFormatContext> fmtctxt,
    const std::string& codecname,
    size_t height, size_t width,
    float framerate, float bitrate, size_t gop,
    AVCodec*& codec) {

  AVStream* retval = avformat_new_stream(fmtctxt.get(), codec);

  if (!retval) {
    boost::format m("ffmpeg::avformat_new_stream(format=`%s' == `%s', codec=`%s[0x%x]' == `%s') failed: could not allocate video stream container for encoding video to file `%s'");
    m % fmtctxt->oformat->name % fmtctxt->oformat->long_name 
      % codec->id % codec->name % codec->long_name % filename.string();
    throw std::runtime_error(m.str());
  }

  /* Some adjustments on the newly created stream */
  retval->id = fmtctxt->nb_streams-1; ///< this should be 0, normally
 
  /* Set user parameters */
  avcodec_get_context_defaults3(retval->codec, codec);
  retval->codec->codec_id = codec->id;

  retval->codec->bit_rate = bitrate;

  /* Resolution must be a multiple of two. */
  if (height%2 != 0 || height == 0 || width%2 != 0 || width == 0) {
    boost::format m("ffmpeg only accepts video height and width if they are, both, multiples of two, but you supplied %d x %d while configuring video output for file `%s' - correct these and re-run");
    m % height % width % filename.string();
    deallocate_stream(retval);
    throw std::runtime_error(m.str());
  }
  retval->codec->width    = width;
  retval->codec->height   = height;

  /* timebase: This is the fundamental unit of time (in seconds) in terms
   * of which frame timestamps are represented. For fixed-fps content,
   * timebase should be 1/framerate and timestamp increments should be
   * identical to 1. */
  retval->codec->time_base.den = framerate;
  retval->codec->time_base.num = 1;
  retval->codec->gop_size      = gop; /* emit one intra frame every X at most */
  retval->codec->pix_fmt       = AV_PIX_FMT_YUV420P;

  if (retval->codec->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
    /* just for testing, we also add B frames */
    retval->codec->max_b_frames = 2;
  }

  if (retval->codec->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
    /* Needed to avoid using macroblocks in which some coeffs overflow.
     * This does not happen with normal video, it just happens here as
     * the motion of the chroma plane does not match the luma plane. */
    retval->codec->mb_decision = 2;
  }

  /* Some formats want stream headers to be separate. */
  if (fmtctxt->oformat->flags & AVFMT_GLOBALHEADER) {
    retval->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
  }

  return retval;
}

#else // region of code if FFMPEG_VERSION_INT < 0.9.0

static AVStream* allocate_stream(
    const boost::filesystem::path& filename,
    boost::shared_ptr<AVFormatContext> fmtctxt,
    const std::string& codecname,
    size_t height, size_t width,
    float framerate, float bitrate, size_t gop,
    AVCodec*& codec) {
  
  AVStream* retval = av_new_stream(fmtctxt.get(), 0);

  if (!retval) {
    boost::format m("ffmpeg::av_new_stream(format=`%s' == `%s') failed: could not allocate video stream container for encoding video to file `%s'");
    m % fmtctxt->oformat->name % fmtctxt->oformat->long_name 
      % filename.string();
    throw std::runtime_error(m.str());
  }

  /* Prepare the stream */
  retval->codec->codec_id = codec->id;
  retval->codec->codec_type = codec->type;

  /* Set user parameters */
  retval->codec->bit_rate = bitrate;

  /* Resolution must be a multiple of two. */
  if (height%2 != 0 || height == 0 || width%2 != 0 || width == 0) {
    boost::format m("ffmpeg only accepts video height and width if they are, both, multiples of two, but you supplied %d x %d while configuring video output for file `%s' - correct these and re-run");
    m % height % width % filename.string();
    deallocate_stream(retval);
    throw std::runtime_error(m.str());
  }
  retval->codec->width    = width;
  retval->codec->height   = height;

  /* timebase: This is the fundamental unit of time (in seconds) in terms
   * of which frame timestamps are represented. For fixed-fps content,
   * timebase should be 1/framerate and timestamp increments should be
   * identical to 1. */
  retval->codec->time_base.den = framerate;
  retval->codec->time_base.num = 1;
  retval->codec->gop_size      = gop; /* emit one intra frame every X at most */
  retval->codec->pix_fmt       = AV_PIX_FMT_YUV420P;

  if (retval->codec->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
    /* just for testing, we also add B frames */
    retval->codec->max_b_frames = 2;
  }

  if (retval->codec->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
    /* Needed to avoid using macroblocks in which some coeffs overflow.
     * This does not happen with normal video, it just happens here as
     * the motion of the chroma plane does not match the luma plane. */
    retval->codec->mb_decision = 2;
  }

  /* Some formats want stream headers to be separate. */
  if (fmtctxt->oformat->flags & AVFMT_GLOBALHEADER) {
    retval->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
  }

  return retval;
}
#endif // FFMPEG_VERSION_INT >= 0.9.0

/**
 * Allocates a frame
 */
static AVFrame* allocate_frame(const boost::filesystem::path& filename, 
    boost::shared_ptr<AVStream> stream, PixelFormat pixfmt=AV_PIX_FMT_NONE) {

  /* allocate and init a re-usable frame */
  AVFrame* retval = avcodec_alloc_frame();
  if (!retval) {
    boost::format m("ffmpeg::avcodec_alloc_frame() failed: cannot allocate frame to start encoding video file `%s'");
    m % filename.string();
    throw std::runtime_error(m.str());
  }

  if (pixfmt == AV_PIX_FMT_NONE) {
    pixfmt = stream->codec->pix_fmt;
  }

  size_t size = avpicture_get_size(pixfmt, stream->codec->width, stream->codec->height);

  uint8_t* picture_buf = (uint8_t*)av_malloc(size);

  if (!picture_buf) {
    av_free(retval);
    boost::format m("ffmpeg::av_malloc(size=%d) failed: cannot picture buffer to start encoding video file `%s'");
    m % size % filename.string();
  }

  /**
   * This method will distributed the allocated memory for "picture_buf" into
   * the several frame "data" pointers. This will make sure that, for example,
   * multi-plane formats have the necessary amount of space pre-set and
   * indicated for each plane in each of the "data" placeholders (4 at total).
   *
   * For example, for the YUV420P format (1 byte per Y value and 0.5 byte per U
   * and V values), it will make sure that the data allocated for picture_buf
   * is split following the correct plane proportion 2:1:1 for each plane
   * Y:U:V.
   *
   * For an RGB24 (packed RGB) format, it will make sure the linesize is set to
   * 3 times the image width so it can pack the R, G and B bytes together in a
   * single line.
   */
  avpicture_fill((AVPicture *)retval, picture_buf, pixfmt,
      stream->codec->width, stream->codec->height);

  return retval;
}

static void deallocate_frame(AVFrame* f) {
  if (f) {
    if (f->data[0]) av_free(f->data[0]);
    av_free(f);
  }
}

/**
 * Handles the software scaler allocation and deallocation
 */
static SwsContext* allocate_swscaler(const boost::filesystem::path& filename,
    boost::shared_ptr<AVStream> stream) {
  /**
   * Initializes the software scaler (SWScale) so we can convert images to
   * the movie native format from RGB. You can define which kind of
   * interpolation to perform. Some options from libswscale are:
   * SWS_FAST_BILINEAR, SWS_BILINEAR, SWS_BICUBIC, SWS_X, SWS_POINT, SWS_AREA
   * SWS_BICUBLIN, SWS_GAUSS, SWS_SINC, SWS_LANCZOS, SWS_SPLINE
   */
  AVCodecContext* ctxt = stream->codec;
  SwsContext* retval = sws_getContext(ctxt->width, ctxt->height, 
      AV_PIX_FMT_RGB24, ctxt->width, ctxt->height, ctxt->pix_fmt, SWS_BICUBIC,
      0, 0, 0);
  if (!retval) {
    boost::format m("ffmpeg::sws_getContext(src_width=%d, src_height=%d, src_pix_format=`%s', dest_width=%d, dest_height=%d, dest_pix_format=`%s', flags=SWS_BICUBIC, 0, 0, 0) failed: cannot get software scaler context to start encoding video file `%s'");
    m % ctxt->width % ctxt->height % av_get_pix_fmt_name(AV_PIX_FMT_RGB24)
      % ctxt->width % ctxt->height % av_get_pix_fmt_name(ctxt->pix_fmt)
      % filename.string();
    throw std::runtime_error(m.str());
  }
  return retval;
}

static void deallocate_swscaler(SwsContext* s) {
  if (s) sws_freeContext(s);
}

/**
 * Allocate and deallocate video buffer for FFmpeg < 0.11.0
 */
static uint8_t* allocate_buffer(boost::shared_ptr<AVFormatContext> format_context) {
  uint8_t* retval=0;

#if FFMPEG_VERSION_INT < 0x000b00 // FFmpeg < 0.11.0

  if (!(format_context->oformat->flags & AVFMT_RAWPICTURE)) {
    /* allocate output buffer */
    /* XXX: API change will be done */
    /* buffers passed into lav* can be allocated any way you prefer,
       as long as they're aligned enough for the architecture, and
       they're freed appropriately (such as using av_free for buffers
       allocated with av_malloc) */
    retval = reinterpret_cast<uint8_t*>(av_malloc(FFMPEG_VIDEO_BUFFER_SIZE));

    if (!retval) {
      boost::format m("ffmpeg::av_malloc(%d) failed: could not allocate video output buffer for encoding");
      m % FFMPEG_VIDEO_BUFFER_SIZE;
      throw std::runtime_error(m.str());
    }
  }

#endif

  return retval;
}

static void deallocate_buffer(uint8_t* p) {
  if (p) av_free(p);
}

bob::io::VideoWriter::VideoWriter(
    const std::string& filename,
    size_t height,
    size_t width,
    float framerate,
    float bitrate,
    size_t gop,
    const std::string& codec,
    const std::string& format) :
  m_filename(filename),
  m_opened(false),
  m_format_context(allocate_format_context(m_filename, format),
      std::ptr_fun(deallocate_format_context)),
  m_codec(find_encoder(m_filename, m_format_context, codec)),
  m_stream(allocate_stream(m_filename, m_format_context, codec, height,
        width, framerate, bitrate, gop, m_codec), 
      std::ptr_fun(deallocate_stream)),
  m_context_frame(allocate_frame(m_filename, m_stream),
      std::ptr_fun(deallocate_frame)),
  m_packed_rgb_frame(allocate_frame(m_filename, m_stream, AV_PIX_FMT_RGB24),
      std::ptr_fun(deallocate_frame)),
  m_swscaler(allocate_swscaler(m_filename, m_stream),
      std::ptr_fun(deallocate_swscaler)),
  m_buffer(allocate_buffer(m_format_context), std::ptr_fun(deallocate_buffer)),
  m_height(height),
  m_width(width),
  m_framerate(framerate),
  m_bitrate(bitrate),
  m_gop(gop),
  m_codecname(codec),
  m_formatname(format),
  m_current_frame(0)
{
  /* Opens the codec context */
  int ok = avcodec_open2(m_stream->codec, m_codec, 0);
  if (ok < 0) {
    boost::format m("ffmpeg::avcodec_open2(codec=`%s'(0x%x) == `%s') failed: cannot open codec to start encoding video file `%s' (error %d)");
    m % m_codec->name % m_codec->id % m_codec->long_name % m_filename.string() % ok;
    throw std::runtime_error(m.str());
  }

  av_dump_format(m_format_context.get(), 0, filename.c_str(), 1);

  /* open the output file, if needed */
  if (!(m_format_context->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&m_format_context->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0) {
      boost::format m("ffmpeg::avio_open(filename=`%s', AVIO_FLAG_WRITE) failed: cannot open output file for writing");
      m % filename.c_str();
      throw std::runtime_error(m.str());
    }
  }

  /* Write the stream header, if any. */
  int error = avformat_write_header(m_format_context.get(), 0);
  if (error < 0) {
    boost::format m("ffmpeg::avformat_write_header(filename=`%s') failed: cannot write header to output file for some reason (error = %d)");
    m % filename.c_str() % error;
    throw std::runtime_error(m.str());
  }

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

void bob::io::VideoWriter::flush_encoder () {

#if FFMPEG_VERSION_INT >= 0x000b00 // FFmpeg version >= 0.11.0
    
  //We only need to flush codecs that have delayed data processing
  if (!(m_codec->capabilities & CODEC_CAP_DELAY)) return;

  while (true) {

    /* encode the image */
    AVPacket pkt;
    av_init_packet(&pkt);

    pkt.data = 0; // packet data will be allocated by the encoder
    pkt.size = 0;

    int got_output;
    int ok = avcodec_encode_video2(m_stream->codec, &pkt, 0, &got_output);

    if (ok < 0) {
      boost::format m("ffmpeg::avcodec_encode_video2() failed: failed to encode video frame (error = %d) while writing to file `%s'");
      m % ok % m_filename.string();
      throw std::runtime_error(m.str());
    }

    /* If size is zero, it means the image was buffered. */
    else if (got_output) {
      if (m_stream->codec->coded_frame->key_frame) pkt.flags |= AV_PKT_FLAG_KEY;
      pkt.stream_index = m_stream->index;

      /* Write the compressed frame to the media file. */
      ok = av_interleaved_write_frame(m_format_context.get(), &pkt);
      if (ok && (ok != AVERROR(EINVAL))) {
        boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to encode video frame (error = %d) while flushing remaining frames to file `%s'");
        m % ok % m_filename.string();
        throw std::runtime_error(m.str());
      }
    }

    /* encoded the video, but no pkt got back => video is flushed */
    else if (ok == 0) break;

  }

#endif
}

void bob::io::VideoWriter::close() {

  /* Flushes the current encoder if necessary */
  flush_encoder();

  /* Write the trailer, if any. The trailer must be written before you
   * close the CodecContexts open when you wrote the header; otherwise
   * av_write_trailer() may try to use memory that was freed on
   * av_codec_close(). */
  int error = av_write_trailer(m_format_context.get());
  if (error < 0) {
    boost::format m("ffmpeg::av_write_trailer(filename=`%s') failed: cannot write trailer to output file for some reason (error = %d)");
    m % m_filename.string() % error;
    throw std::runtime_error(m.str());
  }

  /* Closes the codec context */
  avcodec_close(m_stream->codec);

  /* Destroyes resources in an orderly fashion */
  m_context_frame.reset();
  m_packed_rgb_frame.reset();
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

/**
 * Transforms from Bob's planar 8-bit RGB representation to whatever is
 * required by the FFmpeg encoder output context (peeked from the AVStream
 * object passed).
 *
 * If the destination stream accepts packed RGB, we pack it from Bob's planar
 * representation directly into the output buffer (frame). Otherwise, we have
 * to go through a temporary resource, first pack and then convert to the final
 * representation. Unfortunately, the software scaler in ffmpeg cannot handle a
 * planar 8-bit RGB representation.
 */
static void bob_image_to_context(const blitz::Array<uint8_t,3>& data,
    boost::shared_ptr<AVStream> stream, 
    boost::shared_ptr<SwsContext> scaler,
    boost::shared_ptr<AVFrame> output_frame,
    boost::shared_ptr<AVFrame> tmp_frame) {

  size_t width = stream->codec->width;
  size_t height = stream->codec->height;
  
  if (stream->codec->pix_fmt != PIX_FMT_RGB24) {
    //transpose Bob data from "planar" RGB24 to "packed" RGB24 on temporary
    //frame space
    blitz::Array<uint8_t,3> ordered(tmp_frame->data[0],
        blitz::shape(height, width, 3), blitz::neverDeleteData);
    ordered = const_cast<blitz::Array<uint8_t,3>&>(data).transpose(1,2,0);

    // converts from packed RGB, 24-bits -> what is required by the context
    // using a software scaler
    sws_scale(scaler.get(), tmp_frame->data, tmp_frame->linesize,
        0, height, output_frame->data, output_frame->linesize);
  }
  
  else {
    //transpose Bob data from "planar" RGB24 to "packed" RGB24
    output_frame->linesize[0] = width*3;
    blitz::Array<uint8_t,3> ordered(output_frame->data[0],
        blitz::shape(height, width, 3), blitz::neverDeleteData);
    ordered = const_cast<blitz::Array<uint8_t,3>&>(data).transpose(1,2,0); //organize for ffmpeg
  }

}
 
void bob::io::VideoWriter::write_video_frame (const blitz::Array<uint8_t,3>& data) {

  bob_image_to_context(data, m_stream, m_swscaler, m_context_frame, 
      m_packed_rgb_frame);

  if (m_format_context->oformat->flags & AVFMT_RAWPICTURE) {
    
    /* Raw video case - directly store the picture in the packet */
    AVPacket pkt;
    av_init_packet(&pkt);

    pkt.flags        |= AV_PKT_FLAG_KEY;
    pkt.stream_index  = m_stream->index;
    pkt.data          = m_context_frame->data[0];
    pkt.size          = sizeof(AVPicture);

    int ok = av_interleaved_write_frame(m_format_context.get(), &pkt);
    if (ok && (ok != AVERROR(EINVAL))) {
      boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to write video frame (error = %d) while encoding file `%s'");
      m % ok % m_filename.string();
      throw std::runtime_error(m.str());
    }

  }
#if FFMPEG_VERSION_INT >= 0x000b00 // FFmpeg version >= 0.11.0
  
  else {

    /* encode the image */
    AVPacket pkt;
    av_init_packet(&pkt);

    pkt.data = 0; // packet data will be allocated by the encoder
    pkt.size = 0;

    int got_output;
    int ok = avcodec_encode_video2(m_stream->codec, &pkt, m_context_frame.get(), &got_output);
    if (ok < 0) {
      boost::format m("ffmpeg::avcodec_encode_video2() failed: failed to encode video frame (error = %d) while writing to file `%s'");
      m % ok % m_filename.string();
      throw std::runtime_error(m.str());
    }

    /* If size is zero, it means the image was buffered. */
    if (got_output) {
      if (m_stream->codec->coded_frame->key_frame) pkt.flags |= AV_PKT_FLAG_KEY;
      pkt.stream_index = m_stream->index;

      /* Write the compressed frame to the media file. */
      ok = av_interleaved_write_frame(m_format_context.get(), &pkt);
      if (ok && (ok != AVERROR(EINVAL))) {
        boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to write video frame (error = %d) while encoding file `%s'");
        m % ok % m_filename.string();
        throw std::runtime_error(m.str());
      }

    } 

  }

#else // == #if FFmpeg version < 0.11.0

  else {
    /* encode the image */
    int out_size = avcodec_encode_video(m_stream->codec, 
        m_buffer.get(), FFMPEG_VIDEO_BUFFER_SIZE, m_context_frame.get());

    /* If size is zero, it means the image was buffered. */
    if (out_size > 0) {

      /* encode the image */
      AVPacket pkt;
      av_init_packet(&pkt);

      if ((size_t)m_stream->codec->coded_frame->pts != AV_NOPTS_VALUE)
        pkt.pts = av_rescale_q(m_stream->codec->coded_frame->pts,
            m_stream->codec->time_base, m_stream->time_base);
      if (m_stream->codec->coded_frame->key_frame)
        pkt.flags |= AV_PKT_FLAG_KEY;

      pkt.stream_index = m_stream->index;
      pkt.data         = m_buffer.get();
      pkt.size         = out_size;

      /* Write the compressed frame to the media file. */
      int ok = av_interleaved_write_frame(m_format_context.get(), &pkt);
      if (ok && (ok != AVERROR(EINVAL))) {
        boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to write video frame (error = %d) while encoding file `%s'");
        m % ok % m_filename.string();
        throw std::runtime_error(m.str());
      }

    }
  }

#endif // FFmpeg version >= 0.11.0

  // Ok, update frame counters
  ++m_current_frame;
  m_typeinfo_video.shape[0] += 1;
  m_context_frame->pts += 1;
}

void bob::io::VideoWriter::append(const blitz::Array<uint8_t,4>& data) {
  if (!m_opened) {
    boost::format m("video writer for file `%s' is closed and cannot be written to");
    m % m_filename.string();
    throw std::runtime_error(m.str());
  }

  //checks data specifications
  if (data.extent(1) != 3 || (size_t)data.extent(2) != m_height || 
      (size_t)data.extent(3) != m_width) {
    boost::format m("input data extents for each frame (the last 3 dimensions of your 4D input array = %dx%dx%d) do not conform to expected format (3x%dx%d), while writing data to file `%s'");
    m % data.extent(1) % data.extent(2) % data.extent(3)
      % m_height % m_width % m_filename.string();
    throw std::runtime_error(m.str());
  }

  blitz::Range a = blitz::Range::all();
  for(int i=data.lbound(0); i<(data.extent(0)+data.lbound(0)); ++i) {
    write_video_frame(data(i, a, a, a));
  }
}

void bob::io::VideoWriter::append(const blitz::Array<uint8_t,3>& data) {
  if (!m_opened) {
    boost::format m("video writer for file `%s' is closed and cannot be written to");
    m % m_filename.string();
    throw std::runtime_error(m.str());
  }

  //checks data specifications
  if (data.extent(0) != 3 || (size_t)data.extent(1) != m_height || 
      (size_t)data.extent(2) != m_width) {
    boost::format m("input data extents (%dx%dx%d) do not conform to expected format (3x%dx%d), while writing data to file `%s'");
    m % data.extent(0) % data.extent(1) % data.extent(2)
      % m_height % m_width % m_filename.string();
    throw std::runtime_error(m.str());
  }

  write_video_frame(data);
}

void bob::io::VideoWriter::append(const bob::core::array::interface& data) {
  if (!m_opened) {
    boost::format m("video writer for file `%s' is closed and cannot be written to");
    m % m_filename.string();
    throw std::runtime_error(m.str());
  }

  const bob::core::array::typeinfo& type = data.type();

  if ( type.dtype != bob::core::array::t_uint8 ) {
    boost::format m("input data type = `%s' does not conform to the specified input specifications (3D array = `%s' or 4D array = `%s'), while writing data to file `%s'");
    m % type.str() % m_typeinfo_frame.str() % m_typeinfo_video.str()
      % m_filename.string();
  }

  if ( type.nd == 3 ) { //appends single frame
    if ( (type.shape[0] != 3) || 
        (type.shape[1] != m_height) || 
        (type.shape[2] != m_width) ) {
      boost::format m("input data extents (%dx%dx%d) do not conform to expected format (3x%dx%d), while writing data to file `%s'");
      m % type.shape[0] % type.shape[1] % type.shape[2]
        % m_height % m_width % m_filename.string();
      throw std::runtime_error(m.str());
    }

    blitz::TinyVector<int,3> shape;
    shape = 3, m_height, m_width;
    blitz::Array<uint8_t,3> tmp(const_cast<uint8_t*>(static_cast<const uint8_t*>(data.ptr())), shape,
        blitz::neverDeleteData);
    write_video_frame(tmp);
  }
  
  else if ( type.nd == 4 ) { //appends a sequence of frames
    if ( (type.shape[1] != 3) || 
         (type.shape[2] != m_height) || 
         (type.shape[3] != m_width) ) {
      boost::format m("input data extents for each frame (the last 3 dimensions of your 4D input array = %dx%dx%d) do not conform to expected format (3x%dx%d), while writing data to file `%s'");
      m % type.shape[1] % type.shape[2] % type.shape[3]
        % m_height % m_width % m_filename.string();
      throw std::runtime_error(m.str());
    }
    
    blitz::TinyVector<int,3> shape;
    shape = 3, m_height, m_width;
    unsigned long int frame_size = 3 * m_height * m_width;
    uint8_t* ptr = const_cast<uint8_t*>(static_cast<const uint8_t*>(data.ptr()));

    for(size_t i=0; i<type.shape[0]; ++i) {
      blitz::Array<uint8_t,3> tmp(ptr, shape, blitz::neverDeleteData);
      write_video_frame(tmp);
      ptr += frame_size;
    }
  }

  else {
    boost::format m("input data type information = `%s' does not conform to the specified input specifications (3D array = `%s' or 4D array = `%s'), while writing data to file `%s'");
    m % type.str() % m_typeinfo_frame.str() % m_typeinfo_video.str()
      % m_filename.string();
  }

}
