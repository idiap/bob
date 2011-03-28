/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun 27 Mar 16:10:24 2011 
 *
 * @brief Implementation of a video writer based on ffmpeg, from example output
 * program:
 * http://cekirdek.pardus.org.tr/~ismail/ffmpeg-docs/output-example_8c-source.html.
 */

#include <boost/format.hpp>
#include <boost/preprocessor.hpp>

#include "database/VideoWriter.h"
#include "database/VideoException.h"

namespace db = Torch::database;

db::VideoWriter::VideoWriter(const std::string& filename, size_t height,
    size_t width, float bitrate, float framerate, size_t gop):
  m_filename(filename),
  m_height(height),
  m_width(width),
  m_bitrate(bitrate),
  m_framerate(framerate),
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
  //opens the video file and wait for user input

  if (m_height%2 != 0 || m_width%2 != 0) {
    throw db::FFmpegException(m_filename.c_str(), "height and width must be multiples of 2");
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
    throw db::FFmpegException(m_filename.c_str(), "could not find suitable output format");
  }

  // allocates the output media context
  m_format_ctxt = avformat_alloc_context();
  if (!m_format_ctxt) {
    throw db::FFmpegException(m_filename.c_str(), "could not allocate the output format context");
  }
  m_format_ctxt->oformat = m_oformat_ctxt;

  // if I could not get a codec ID, raise an exception
  if (m_oformat_ctxt->video_codec == CODEC_ID_NONE)
    throw db::FFmpegException(m_filename.c_str(), "could not find suitable codec for encoding");

  // sets the codecnames (short and long versions)
  AVCodec* codec = avcodec_find_decoder(m_oformat_ctxt->video_codec);
  m_codecname = codec->name;
  m_codecname_long = codec->long_name;

  // adds the video stream using the default format codec and initializes it
  m_video_stream = add_video_stream();

  // sets the output parameters (must be done even if no parameters).
  if (av_set_parameters(m_format_ctxt, NULL) < 0) {
    throw db::FFmpegException(m_filename.c_str(), "invalid output parameters");
  }

  dump_format(m_format_ctxt, 0, m_filename.c_str(), 1);

  // now that all the parameters are set, we can open the video codecs
  // and allocate the necessary encode buffers
  open_video();

  // opens the output file, if needed
  if (!(m_oformat_ctxt->flags & AVFMT_NOFILE)) {
    if (url_fopen(&m_format_ctxt->pb, m_filename.c_str(), URL_WRONLY) < 0) {
      throw db::FFmpegException(m_filename.c_str(), "cannot open file");
    }
  }

  // writes the stream header, if any
  av_write_header(m_format_ctxt);
  
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
    throw db::FFmpegException(m_filename.c_str(), "cannot initialize software scaler");
  }
}

db::VideoWriter::~VideoWriter() {
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
  if (m_oformat_ctxt != 0 && !(m_oformat_ctxt->flags & AVFMT_NOFILE)) url_fclose(m_format_ctxt->pb);

  // frees the stream
  if (m_format_ctxt != 0) av_free(m_format_ctxt);
  
  // closes the sw scaler context
  if (m_sws_context) { 
    sws_freeContext(m_sws_context);
    m_sws_context=0; 
  }
  
}

AVStream* db::VideoWriter::add_video_stream() {
  AVCodecContext *c;
  AVStream *st;

  st = av_new_stream(m_format_ctxt, 0);
  if (!st) 
    throw db::FFmpegException(m_filename.c_str(), "cannot allocate stream");

  c = st->codec;
  c->codec_id = m_oformat_ctxt->video_codec;
  c->codec_type = CODEC_TYPE_VIDEO;

  // puts sample parameters
  c->bit_rate = m_bitrate;
  // resolution must be a multiple of two
  c->width = m_width;
  c->height = m_height;
  // time base: this is the fundamental unit of time (in seconds) in terms
  //      of which frame timestamps are represented. for fixed-fps content,
  //	timebase should be 1/framerate and timestamp increments should be
  //	identically 1.
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

AVFrame* db::VideoWriter::alloc_picture(enum PixelFormat pix_fmt) {
  AVFrame* picture = avcodec_alloc_frame();
  
  if (!picture) 
    throw db::FFmpegException(m_filename.c_str(), "cannot allocate frame");

  size_t size = avpicture_get_size(pix_fmt, m_width, m_height);

  uint8_t* picture_buf = (uint8_t*)av_malloc(size);

  if (!picture_buf) {
    av_free(picture);
    throw db::FFmpegException(m_filename.c_str(), "cannot allocate frame buffer");
  }

  avpicture_fill((AVPicture *)picture, picture_buf, pix_fmt, m_width, m_height);
  return picture;
}

void db::VideoWriter::open_video() {
  AVCodec *codec;
  AVCodecContext *c;

  c = m_video_stream->codec;

  // finds the video encoder
  codec = avcodec_find_encoder(c->codec_id);
  if (!codec) throw db::FFmpegException(m_filename.c_str(), "codec not found");

  // opens the codec
  if (avcodec_open(c, codec) < 0) {
    throw db::FFmpegException(m_filename.c_str(), "cannot open codec");
  }

  video_outbuf = 0;
  if (!(m_format_ctxt->oformat->flags & AVFMT_RAWPICTURE)) {
    // allocate output buffer
    // XXX: API change will be done
    //	buffers passed into lav* can be allocated any way you prefer,
    //	as long as they're aligned enough for the architecture, and
    //	they're freed appropriately (such as using av_free for buffers
    //	allocated with av_malloc)
    video_outbuf_size = 200000;
    video_outbuf = (uint8_t*)av_malloc(video_outbuf_size);

    if (!video_outbuf) 
      throw db::FFmpegException(m_filename.c_str(),
          "cannot allocate output buffer");
  }

  // allocates the encoded raw picture
  picture = alloc_picture(c->pix_fmt);

  // if the output format is not RGB24, then a temporary RGB24
  // picture is needed too. It is then converted to the required
  // output format
  if (c->pix_fmt != PIX_FMT_RGB24) tmp_picture = alloc_picture(PIX_FMT_RGB24);
}

/**
 * Very chaotic implementation extracted from the old Video implementation in
 * Torch. Seems to work, but is probably doing too many memcpy's.
 */
void db::VideoWriter::write_video_frame(const blitz::Array<uint8_t,3>& data) {
  int out_size, ret;
  AVCodecContext *c;

  c = m_video_stream->codec;

  if (false) { //m_current_frame >= STREAM_NB_FRAMES)
    // no more frame to compress. The codec has a latency of a few
    //     	frames if using B frames, so we get the last frames by
    //	passing the same picture again
  }
  else {
    if (c->pix_fmt != PIX_FMT_RGB24) {
      // replace data in the buffer frame by the pixmap to encode
      tmp_picture->linesize[0] = c->width*3;
      blitz::Array<uint8_t,3> ordered(tmp_picture->data[0],
          blitz::shape(c->height, c->width, 3), blitz::neverDeleteData);
      ordered = data.transpose(1,2,0); //organize for ffmpeg
      sws_scale(m_sws_context, tmp_picture->data, tmp_picture->linesize,
          0, c->height, picture->data, picture->linesize);
    }
    else {
      picture->linesize[0] = c->width*3;
      blitz::Array<uint8_t,3> ordered(picture->data[0],
          blitz::shape(c->height, c->width, 3), blitz::neverDeleteData);
      ordered = data.transpose(1,2,0); //organize for ffmpeg
    }
  }

  if (m_format_ctxt->oformat->flags & AVFMT_RAWPICTURE) {
    // raw video case. The API will change slightly in the near future for that
    AVPacket pkt;
    av_init_packet(&pkt);

    pkt.flags |= PKT_FLAG_KEY;
    pkt.stream_index= m_video_stream->index;
    pkt.data= (uint8_t *)picture;
    pkt.size= sizeof(AVPicture);

    ret = av_interleaved_write_frame(m_format_ctxt, &pkt);
  }
  else {
    // encodes the image
    out_size = avcodec_encode_video(c, video_outbuf, video_outbuf_size, picture);
    // if zero size, it means the image was buffered
    if (out_size > 0) {
      AVPacket pkt;
      av_init_packet(&pkt);

      if (static_cast<uint64_t>(c->coded_frame->pts) != AV_NOPTS_VALUE)
        pkt.pts= av_rescale_q(c->coded_frame->pts, c->time_base, m_video_stream->time_base);
      if(c->coded_frame->key_frame)
        pkt.flags |= PKT_FLAG_KEY;
      pkt.stream_index= m_video_stream->index;
      pkt.data= video_outbuf;
      pkt.size= out_size;

      // writes the compressed frame in the media file
      ret = av_interleaved_write_frame(m_format_ctxt, &pkt);
    }
    else {
      ret = 0;
    }
  }
  if (ret != 0) {
    throw db::FFmpegException(m_filename.c_str(), "error writing video frame");
  }

  // OK
  ++m_current_frame;
}

void db::VideoWriter::close_video() {
  avcodec_close(m_video_stream->codec);
  av_free(picture->data[0]);
  av_free(picture);
  if (tmp_picture) {
    av_free(tmp_picture->data[0]);
    av_free(tmp_picture);
  }
  av_free(video_outbuf);
}

void db::VideoWriter::append(const blitz::Array<uint8_t,3>& data) {
  //checks data specifications
  if (data.extent(0) != 3 || (size_t)data.extent(1) != m_height || 
      (size_t)data.extent(2) != m_width) {
    throw db::FFmpegException(m_filename.c_str(), 
        "input data does not conform with video specifications");
  }
  write_video_frame(data);
}

void db::VideoWriter::append(const blitz::Array<uint8_t,4>& data) {
  //checks data specifications
  if (data.extent(1) != 3 || (size_t)data.extent(2) != m_height || 
      (size_t)data.extent(3) != m_width) {
    throw db::FFmpegException(m_filename.c_str(), 
        "input data does not conform with video specifications");
  }

  blitz::Range a = blitz::Range::all();
  for(int i=data.lbound(0); i<(data.extent(0)+data.lbound(0)); ++i) {
    write_video_frame(data(i, a, a, a));
  }
}

std::string db::VideoWriter::info() const {
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
