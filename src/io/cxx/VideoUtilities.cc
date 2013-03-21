/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 26 Nov 17:34:12 2012
 *
 * @brief A set of methods to grab information from ffmpeg.
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

#include <set>
#include <boost/token_iterator.hpp>
#include <boost/format.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#if LIBAVUTIL_VERSION_INT >= 0x320f01 //50.15.1 @ ffmpeg-0.6
#  include <libavutil/opt.h>
#  include <libavutil/pixdesc.h>
#endif
#include <libavutil/avstring.h>
#include <libavutil/mathematics.h>
}

#include "bob/io/VideoUtilities.h"
#include "bob/core/logging.h"
#include "bob/config.h"

/**
 * Some code to account for older versions of ffmpeg
 */
#ifndef AV_CODEC_ID_NONE
#define AV_CODEC_ID_NONE CODEC_ID_NONE
#define AV_CODEC_ID_MPEG1VIDEO CODEC_ID_MPEG1VIDEO
#define AV_CODEC_ID_MPEG2VIDEO CODEC_ID_MPEG2VIDEO
#define AV_CODEC_ID_MJPEG CODEC_ID_MJPEG
typedef CodecID AVCodecID;
#endif

#ifndef AV_PKT_FLAG_KEY
#define AV_PKT_FLAG_KEY PKT_FLAG_KEY
#endif

#ifndef AV_PIX_FMT_YUV420P
#define AV_PIX_FMT_YUV420P PIX_FMT_YUV420P
#endif

#if LIBAVCODEC_VERSION_INT < 0x347a00 //52.122.0 @ ffmpeg-0.7
#define AVMEDIA_TYPE_VIDEO CODEC_TYPE_VIDEO
#endif

namespace ffmpeg = bob::io::detail::ffmpeg;

static bool FFMPEG_INITIALIZED = false;

/**
 * Tries to find an encoder name through a decoder 
 */
static AVCodec* try_find_through_decoder(const char* codecname) {
  AVCodec* tmp = avcodec_find_decoder_by_name(codecname);
  if (tmp) return avcodec_find_encoder(tmp->id);
  return 0;
}

/**
 * Returns a list of available codecs from what I wish to support
 */
static void check_codec_support(std::map<std::string, const AVCodec*>& retval) {

  std::string tmp[] = {
    "libvpx",
    "vp8",
    "wmv1",
    "wmv2",
    //"wmv3", /* no encoding support */
    "mjpeg",
    "mpegvideo", // the same as mpeg2video
    "mpeg1video", 
    //"mpeg1video_vdpau", //hw accelerated mpeg1video decoding
    "mpeg2video", // the same as mpegvideo
    //"mpegvideo_vdpau", //hw accelerated mpegvideo decoding
    "mpeg4",
    "msmpeg4",
    //"msmpeg4v1", /* no encoding support */
    "msmpeg4v2", // the same as msmpeg4
    "ffv1",
#if LIBAVUTIL_VERSION_INT >= 0x320f01 //50.15.1 @ ffmpeg-0.6
    //"h263p", //bogus on libav-0.8.4
    "h264",
    //"h264_vdpau", //hw accelerated h264 decoding
    //"theora", //buggy on some platforms
    //"libtheora", //buggy on some platforms
    "libx264",
    "zlib",
#endif
  };

  std::set<std::string> wishlist(tmp, tmp + (sizeof(tmp)/sizeof(tmp[0])));

  if (!FFMPEG_INITIALIZED) {
    /* Initialize libavcodec, and register all codecs and formats. */
    av_log_set_level(AV_LOG_QUIET);
    av_register_all();
    FFMPEG_INITIALIZED = true;
  }

  for (AVCodec* it = av_codec_next(0); it != 0; it = av_codec_next(it) ) {
    if (wishlist.find(it->name) == wishlist.end()) continue; ///< ignore this codec
    if (it->type == AVMEDIA_TYPE_VIDEO) {
      auto exists = retval.find(std::string(it->name));
      if (exists != retval.end() && exists->second->id != it->id) {
        bob::core::warn << "Not overriding video codec \"" << it->long_name 
          << "\" (" << it->name << ")" << std::endl;
      }
      else {
        // a codec is potentially available, check encoder and decoder
        bool has_decoder = (bool)(avcodec_find_decoder(it->id));
        bool has_encoder = (bool)(avcodec_find_encoder(it->id));
        if (!has_encoder) {
          has_encoder = (bool)try_find_through_decoder(it->name);
        }
        if (has_encoder && has_decoder) retval[it->name] = it;
        // else, skip this one (cannot test encoding loop)
      }
    }
  }
}

static void check_iformat_support(std::map<std::string, AVInputFormat*>& retval) {
  
  std::string tmp[] = {
    "avi",
    "mov",
    "flv",
    "mp4",
  };

  std::set<std::string> wishlist(tmp, tmp + (sizeof(tmp)/sizeof(tmp[0])));

  if (!FFMPEG_INITIALIZED) {
    /* initialize libavcodec, and register all codecs and formats. */
    av_log_set_level(AV_LOG_QUIET);
    av_register_all();
    FFMPEG_INITIALIZED = true;
  }

  for (AVInputFormat* it = av_iformat_next(0); it != 0; it = av_iformat_next(it) ) {
    std::vector<std::string> names;
    ffmpeg::tokenize_csv(it->name, names);
    for (auto k = names.begin(); k != names.end(); ++k) {
      if (wishlist.find(*k) == wishlist.end()) continue; ///< ignore this format
      auto exists = retval.find(*k);
      if (exists != retval.end()) {
        bob::core::warn << "Not overriding input video format \"" 
          << it->long_name << "\" (" << *k 
          << ") which is already assigned to \"" << exists->second->long_name 
          << "\"" << std::endl;
      }
      else retval[*k] = it;
    }
  }
}

static void check_oformat_support(std::map<std::string, AVOutputFormat*>& retval) {
  
  std::string tmp[] = {
    "avi",
    "mov",
    "flv",
    "mp4",
  };

  std::set<std::string> wishlist(tmp, tmp + (sizeof(tmp)/sizeof(tmp[0])));

  if (!FFMPEG_INITIALIZED) {
    /* initialize libavcodec, and register all codecs and formats. */
    av_log_set_level(AV_LOG_QUIET);
    av_register_all();
    FFMPEG_INITIALIZED = true;
  }

  for (AVOutputFormat* it = av_oformat_next(0); it != 0; it = av_oformat_next(it) ) {
    std::vector<std::string> names;
    ffmpeg::tokenize_csv(it->name, names);
    for (auto k = names.begin(); k != names.end(); ++k) {
      if (wishlist.find(*k) == wishlist.end()) continue; ///< ignore this format
      auto exists = retval.find(*k);
      if (exists != retval.end()) {
        bob::core::warn << "Not overriding input video format \"" 
          << it->long_name << "\" (" << *k 
          << ") which is already assigned to \"" << exists->second->long_name 
          << "\"" << std::endl;
      }
      else retval[*k] = it;
    }
  }
}

/**
 * Defines which combinations of codecs and output formats are valid.
 */
static void define_output_support_map(std::map<AVOutputFormat*, std::vector<const AVCodec*> >& retval) {
  std::map<std::string, const AVCodec*> cdict;
  check_codec_support(cdict);
  std::map<std::string, AVOutputFormat*> odict;
  check_oformat_support(odict);

  auto it = odict.find("avi");
  if (it != odict.end()) { // ".avi" format is available
    retval[it->second].clear();
    for (auto jt = cdict.begin(); jt != cdict.end(); ++jt) {
      retval[it->second].push_back(jt->second); // all formats are possible
    }
  }
  
  it = odict.find("mov");
  if (it != odict.end()) { // ".mov" format is available
    retval[it->second].clear();
    for (auto jt = cdict.begin(); jt != cdict.end(); ++jt) {
      retval[it->second].push_back(jt->second); // all formats are possible
    }
  }
  
  it = odict.find("flv");
  if (it != odict.end()) { // ".flv" format is available
    retval[it->second].clear();
    std::string tmp[] = {
      "libx264", 
      "h264", 
      //"h264_vdpau"
    };
    std::vector<std::string> codecs(tmp, tmp + (sizeof(tmp)/sizeof(tmp[0])));
    for (auto jt = codecs.begin(); jt != codecs.end(); ++jt) {
      auto kt = cdict.find(*jt);
      if (kt != cdict.end()) retval[it->second].push_back(kt->second);
    }
  }

  it = odict.find("mp4");
  if (it != odict.end()) { // ".mp4" format is available
    retval[it->second].clear();
    std::string tmp[] = {
      "libx264",
      "h264", 
      //"h264_vdpau", 
      "mjpeg", 
      "mpeg1video", 
      //"mpegvideo_vdpau"
      "mpeg2video",
      "mpegvideo",
      "mpeg4",
    };
    std::vector<std::string> codecs(tmp, tmp + (sizeof(tmp)/sizeof(tmp[0])));
    for (auto jt = codecs.begin(); jt != codecs.end(); ++jt) {
      auto kt = cdict.find(*jt);
      if (kt != cdict.end()) retval[it->second].push_back(kt->second);
    }
  }
}

void ffmpeg::tokenize_csv(const char* what, std::vector<std::string>& values) {
  if (!what) return;
  boost::char_separator<char> sep(",");
  std::string w(what);
  boost::tokenizer< boost::char_separator<char> > tok(w, sep);
  for (auto k = tok.begin(); k != tok.end(); ++k) values.push_back(*k);
}

void ffmpeg::codecs_installed (std::map<std::string, const AVCodec*>& installed) {
  for (AVCodec* it = av_codec_next(0); it != 0; it = av_codec_next(it) ) {
    if (it->type == AVMEDIA_TYPE_VIDEO) {
      /**
      auto exists = installed.find(std::string(it->name));
      if (exists != installed.end() && exists->second->id != it->id) {
        bob::core::warn << "Not overriding video codec \"" << it->long_name 
          << "\" (" << it->name << ")" << std::endl;
      }
      else **/
      installed[it->name] = it;
    }
  }
}

void ffmpeg::codecs_supported (std::map<std::string, const AVCodec*>& installed) {
  check_codec_support(installed);
}

bool ffmpeg::codec_is_supported (const std::string& name) {
  std::map<std::string, const AVCodec*> cdict;
  ffmpeg::codecs_supported(cdict);
  return (cdict.find(name) != cdict.end());
}

void ffmpeg::iformats_supported (std::map<std::string, AVInputFormat*>& installed) {
  check_iformat_support(installed);
}

bool ffmpeg::iformat_is_supported (const std::string& name) {
  std::map<std::string, AVInputFormat*> idict;
  ffmpeg::iformats_supported(idict);
  std::vector<std::string> names;
  ffmpeg::tokenize_csv(name.c_str(), names);
  for (auto k = names.begin(); k != names.end(); ++k) {
    if (idict.find(*k) != idict.end()) return true;
  }
  return false;
}

void ffmpeg::iformats_installed (std::map<std::string, AVInputFormat*>& installed) {
  for (AVInputFormat* it = av_iformat_next(0); it != 0; it = av_iformat_next(it) ) {
    std::vector<std::string> names;
    ffmpeg::tokenize_csv(it->name, names);
    for (auto k = names.begin(); k != names.end(); ++k) {
      auto exists = installed.find(*k);
      if (exists != installed.end()) {
        bob::core::warn << "Not overriding input video format \"" 
          << it->long_name << "\" (" << *k 
          << ") which is already assigned to \"" << exists->second->long_name 
          << "\"" << std::endl;
      }
      else installed[*k] = it;
    }
  }
}

void ffmpeg::oformats_supported (std::map<std::string, AVOutputFormat*>& installed) {
  check_oformat_support(installed);
}

bool ffmpeg::oformat_is_supported (const std::string& name) {
  std::map<std::string, AVOutputFormat*> odict;
  ffmpeg::oformats_supported(odict);
  std::vector<std::string> names;
  ffmpeg::tokenize_csv(name.c_str(), names);
  for (auto k = names.begin(); k != names.end(); ++k) {
    if (odict.find(*k) != odict.end()) return true;
  }
  return false;
}

void ffmpeg::oformats_installed (std::map<std::string, AVOutputFormat*>& installed) {
  for (AVOutputFormat* it = av_oformat_next(0); it != 0; it = av_oformat_next(it) ) {
    if (!it->video_codec) continue;
    std::vector<std::string> names;
    ffmpeg::tokenize_csv(it->name, names);
    for (auto k = names.begin(); k != names.end(); ++k) {
      auto exists = installed.find(*k);
      if (exists != installed.end()) {
        bob::core::warn << "Not overriding output video format \""
          << it->long_name << "\" (" << *k 
          << ") which is already assigned to \"" << exists->second->long_name 
          << "\"" << std::endl;
      }
      else installed[*k] = it;
    }
  }
}

void ffmpeg::oformat_supported_codecs (const std::string& name,
    std::vector<const AVCodec*>& installed) {
  std::map<AVOutputFormat*, std::vector<const AVCodec*> > format2codec;
  define_output_support_map(format2codec);
  std::map<std::string, AVOutputFormat*> odict;
  ffmpeg::oformats_supported(odict);
  auto it = odict.find(name);
  if (it == odict.end()) {
    boost::format f("output format `%s' is not supported by this build");
    f % name;
    throw std::runtime_error(f.str());
  }
  installed = format2codec[it->second];
}

bool ffmpeg::oformat_supports_codec (const std::string& name,
    const std::string& codecname) {
  std::vector<const AVCodec*> codecs;
  oformat_supported_codecs(name, codecs);
  for (auto k=codecs.begin(); k!=codecs.end(); ++k) {
    if (codecname == (*k)->name) return true;
  }
  return false;
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

static void deallocate_input_format_context(AVFormatContext* c) {
# if LIBAVFORMAT_VERSION_INT < 0x351500 //53.21.0 @ ffmpeg-0.9 + libav-0.8.4

  av_close_input_file(c);

# else

  avformat_close_input(&c);

# endif
}

boost::shared_ptr<AVFormatContext> ffmpeg::make_input_format_context(
    const std::string& filename) {

  AVFormatContext* retval = 0;

# if LIBAVFORMAT_VERSION_INT >= 0x346e00 //52.110.0 @ ffmpeg-0.7
  
  int ok = avformat_open_input(&retval, filename.c_str(), 0, 0);
  if (ok != 0) {
    boost::format m("ffmpeg::avformat_open_input(filename=`%s') failed: ffmpeg reported %d == `%s'");
    m % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

# else
  
  int ok = av_open_input_file(&retval, filename.c_str(), 0, 0, 0);
  if (ok != 0) {
    boost::format m("ffmpeg::av_open_input_file(filename=`%s') failed: ffmpeg reported %d == `%s'");
    m % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

# endif

  // creates and protects the return value
  boost::shared_ptr<AVFormatContext> shared_retval(retval, std::ptr_fun(deallocate_input_format_context));

  // retrieve stream information, throws if cannot find it

# if LIBAVFORMAT_VERSION_INT >= 0x350400 //53.4.0 @ ffmpeg-0.8

  ok = avformat_find_stream_info(retval, 0);
  
  if (ok < 0) {
    boost::format m("ffmpeg::avformat_find_stream_info(filename=`%s') failed: ffmpeg reported %d == `%s'");
    m % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

# else

  ok = av_find_stream_info(retval);

  if (ok < 0) {
    boost::format m("ffmpeg::av_find_stream_info(filename=`%s') failed: ffmpeg reported %d == `%s'");
    m % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

# endif

  return shared_retval;
}
  
int ffmpeg::find_video_stream(const std::string& filename, boost::shared_ptr<AVFormatContext> format_context) {

# if LIBAVFORMAT_VERSION_INT >= 0x346e00 //52.110.0 @ ffmpeg-0.7

  int retval = av_find_best_stream(format_context.get(), AVMEDIA_TYPE_VIDEO,
      -1, -1, 0, 0);

  if (retval < 0) {
    boost::format m("ffmpeg::av_find_stream_info(`%s') failed: cannot find any video streams on this file - ffmpeg reports error %d == `%s'");
    m % filename % retval % ffmpeg_error(retval);
    throw std::runtime_error(m.str());
  }

  return retval;

# else

  // Look for the first video stream in the file
  int retval = -1;
  
  for (size_t i=0; i<format_context->nb_streams; ++i) {
    if (format_context->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO) {
      retval = i;
      break;
    }
  }

  if(retval == -1) {
    boost::format m("error opening video file `%s': cannot find any video streams on this file (looked at all %d streams by iterative search)");
    m % filename % format_context->nb_streams;
    throw std::runtime_error(m.str());
  }

  return retval;

# endif

}
  
AVCodec* ffmpeg::find_decoder(const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context, int stream_index) {

  AVCodec* retval = avcodec_find_decoder(format_context->streams[stream_index]->codec->codec_id);

  if (!retval) {
    boost::format m("ffmpeg::avcodec_find_decoder(0x%x) failed: cannot find a suitable codec to read stream %d of file `%s'");
    m % format_context->streams[stream_index]->codec->codec_id 
      % stream_index % filename;
    throw std::runtime_error(m.str());
  }

  return retval;
}

#ifndef HAVE_FFMPEG_AVFORMAT_ALLOC_OUTPUT_CONTEXT2

/**
 * This method was copied from ffmpeg-0.8 and is used in case it is not defined
 * on older versions of ffmpeg, for its convenience.
 */
static int avformat_alloc_output_context2(AVFormatContext **avctx,
    AVOutputFormat *oformat, const char *format, const char *filename) {

  AVFormatContext *s = avformat_alloc_context();
  int ret = 0;

  *avctx = NULL;
  if (!s)
    goto nomem;

  if (!oformat) {
    if (format) {
#if LIBAVFORMAT_VERSION_INT >= 0x344002 // 52.64.2 @ ffmpeg-0.6
      oformat = av_guess_format(format, NULL, NULL);
#else
      oformat = guess_format(format, NULL, NULL);
#endif
      if (!oformat) {
        av_log(s, AV_LOG_ERROR, "Requested output format '%s' is not a suitable output format\n", format);
        ret = AVERROR(EINVAL);
        goto error;
      }
    } else {
#if LIBAVFORMAT_VERSION_INT >= 0x344002 // 52.64.2 @ ffmpeg-0.6
      oformat = av_guess_format(NULL, filename, NULL);
#else
      oformat = guess_format(NULL, filename, NULL);
#endif
      if (!oformat) {
        ret = AVERROR(EINVAL);
        av_log(s, AV_LOG_ERROR, "Unable to find a suitable output format for '%s'\n",
            filename);
        goto error;
      }
    }
  }

  s->oformat = oformat;

#if LIBAVFORMAT_VERSION_INT >= 0x344002 // 52.64.2 @ ffmpeg-0.6
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
#endif

  if (filename)
    av_strlcpy(s->filename, filename, sizeof(s->filename));
  *avctx = s;
  return 0;
nomem:
  av_log(s, AV_LOG_ERROR, "Out of memory\n");
  ret = AVERROR(ENOMEM);
error:
#if LIBAVFORMAT_VERSION_INT >= 0x344002 // 52.64.2 @ ffmpeg-0.6
  avformat_free_context(s);
#else
  av_free(s);
#endif
  return ret;

}

#endif

static void deallocate_output_format_context(AVFormatContext* f) {
  if (f) av_free(f);
}

boost::shared_ptr<AVFormatContext> ffmpeg::make_output_format_context(
    const std::string& filename, const std::string& formatname) {

  AVFormatContext* retval;
  const char* filename_c = filename.c_str();
  const char* formatname_c = formatname.c_str();

  if (formatname.size() != 0) {
    int ok = avformat_alloc_output_context2(&retval, 0, formatname_c, 
        filename_c);
    if (ok < 0) {
      boost::format m("ffmpeg::avformat_alloc_output_context2() failed: could not allocate output context based on format name == `%s', filename == `%s' - ffmpeg reports error %d == `%s'");
      m % formatname_c % filename_c % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
  }
  else {
    int ok = avformat_alloc_output_context2(&retval, 0, 0, filename_c);
    if (ok < 0) {
      boost::format m("ffmpeg::avformat_alloc_output_context2() failed: could not allocate output context based only on filename == `%s' - ffmpeg reports error %d == `%s'");
      m % formatname_c % filename_c % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
  }

  return boost::shared_ptr<AVFormatContext>(retval, std::ptr_fun(deallocate_output_format_context));
}

AVCodec* ffmpeg::find_encoder(const std::string& filename,
    boost::shared_ptr<AVFormatContext> fmtctxt, const std::string& codecname) {

  AVCodec* retval = 0;

  /* find the video encoder */
  if (codecname.size() != 0) {
    retval = avcodec_find_encoder_by_name(codecname.c_str());
    if (!retval) retval = try_find_through_decoder(codecname.c_str());
    if (!retval) {
      boost::format m("ffmpeg::avcodec_find_encoder_by_name(`%s') failed: could not find a suitable codec for encoding video file `%s' using the output format `%s' == `%s'");
      m % codecname % filename % fmtctxt->oformat->name 
        % fmtctxt->oformat->long_name;
      throw std::runtime_error(m.str());
    }
  }
  else {
    if (fmtctxt->oformat->video_codec == AV_CODEC_ID_NONE) {
      boost::format m("could not identify codec for encoding video file `%s'; tried codec with name `%s' first and then tried output format's `%s' == `%s' video_codec entry, which was also null");
      m % filename % fmtctxt->oformat->name % fmtctxt->oformat->long_name;
      throw std::runtime_error(m.str());
    }
    retval = avcodec_find_encoder(fmtctxt->oformat->video_codec);

    if (!retval) {
      boost::format m("ffmpeg::avcodec_find_encoder(0x%x) failed: could not find encoder for codec with identifier for encoding video file `%s' using the output format `%s' == `%s'");
      m % fmtctxt->oformat->video_codec % filename
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

boost::shared_ptr<AVStream> ffmpeg::make_stream(
    const std::string& filename,
    boost::shared_ptr<AVFormatContext> fmtctxt,
    const std::string& codecname,
    size_t height, size_t width,
    float framerate, float bitrate, size_t gop,
    AVCodec* codec) {

#if LIBAVFORMAT_VERSION_INT >= 0x351500 //53.21.0 @ ffmpeg-0.9

  AVStream* retval = avformat_new_stream(fmtctxt.get(), codec);

  if (!retval) {
    boost::format m("ffmpeg::avformat_new_stream(format=`%s' == `%s', codec=`%s[0x%x]' == `%s') failed: could not allocate video stream container for encoding video to file `%s'");
    m % fmtctxt->oformat->name % fmtctxt->oformat->long_name 
      % codec->id % codec->name % codec->long_name % filename;
    throw std::runtime_error(m.str());
  }

  /* Set user parameters */
  avcodec_get_context_defaults3(retval->codec, codec);

#else // region of code if FFMPEG_VERSION_INT < 0.9.0

  AVStream* retval = av_new_stream(fmtctxt.get(), 0);

  if (!retval) {
    boost::format m("ffmpeg::av_new_stream(format=`%s' == `%s') failed: could not allocate video stream container for encoding video to file `%s'");
    m % fmtctxt->oformat->name % fmtctxt->oformat->long_name % filename;
    throw std::runtime_error(m.str());
  }
  
  /* Prepare the stream */
  retval->codec->codec_type = codec->type;

#endif // LIBAVFORMAT_VERSION_INT >= 53.42.0
  
  /* Some adjustments on the newly created stream */
  retval->id = fmtctxt->nb_streams-1; ///< this should be 0, normally
  
  retval->codec->codec_id = codec->id;
  
  /* Set user parameters */
  retval->codec->bit_rate = bitrate;

  /* Resolution must be a multiple of two. */
  if (height%2 != 0 || height == 0 || width%2 != 0 || width == 0) {
    boost::format m("ffmpeg only accepts video height and width if they are, both, multiples of two, but you supplied %d x %d while configuring video output for file `%s' - correct these and re-run");
    m % height % width % filename;
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
  if (codec->pix_fmts && codec->pix_fmts[0] != -1) {
    retval->codec->pix_fmt     = codec->pix_fmts[0];
  }

# ifdef HAVE_FFMPEG_AVCOLOR_RANGE
  if (retval->codec->codec_id == AV_CODEC_ID_MJPEG) {
    /* set jpeg color range */
    retval->codec->color_range = AVCOL_RANGE_JPEG;
  }
# endif

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

  return boost::shared_ptr<AVStream>(retval, std::ptr_fun(deallocate_stream));
}

static void deallocate_frame(AVFrame* f) {
  if (f) {
    if (f->data[0]) av_free(f->data[0]);
    av_free(f);
  }
}

boost::shared_ptr<AVFrame> ffmpeg::make_frame(const std::string& filename,
    boost::shared_ptr<AVCodecContext> codec, PixelFormat pixfmt) {

  /* allocate and init a re-usable frame */
  AVFrame* retval = avcodec_alloc_frame();
  if (!retval) {
    boost::format m("ffmpeg::avcodec_alloc_frame() failed: cannot allocate frame to start encoding video file `%s'");
    m % filename;
    throw std::runtime_error(m.str());
  }

  size_t size = avpicture_get_size(pixfmt, codec->width, codec->height);

  uint8_t* picture_buf = (uint8_t*)av_malloc(size);

  if (!picture_buf) {
    av_free(retval);
    boost::format m("ffmpeg::av_malloc(size=%d) failed: cannot picture buffer to start reading or writing video file `%s'");
    m % size % filename;
    throw std::runtime_error(m.str());
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
      codec->width, codec->height);

  return boost::shared_ptr<AVFrame>(retval, std::ptr_fun(deallocate_frame));
}

static void deallocate_empty_frame(AVFrame* f) {
  if (f) av_free(f);
}

boost::shared_ptr<AVFrame> ffmpeg::make_empty_frame(const std::string& filename) {
  AVFrame* retval = avcodec_alloc_frame();
  if (!retval) {
    boost::format m("ffmpeg::avcodec_alloc_frame() failed: cannot allocate (empty) frame to start reading video file `%s'");
    m % filename;
    throw std::runtime_error(m.str());
  }
  return boost::shared_ptr<AVFrame>(retval, std::ptr_fun(deallocate_empty_frame));
}

static void deallocate_swscaler(SwsContext* s) {
  if (s) sws_freeContext(s);
}

boost::shared_ptr<SwsContext> ffmpeg::make_scaler
(const std::string& filename, boost::shared_ptr<AVCodecContext> ctxt,
 PixelFormat source_pixel_format, PixelFormat dest_pixel_format) {

  /**
   * Initializes the software scaler (SWScale) so we can convert images to
   * the movie native format from RGB. You can define which kind of
   * interpolation to perform. Some options from libswscale are:
   * SWS_FAST_BILINEAR, SWS_BILINEAR, SWS_BICUBIC, SWS_X, SWS_POINT, SWS_AREA
   * SWS_BICUBLIN, SWS_GAUSS, SWS_SINC, SWS_LANCZOS, SWS_SPLINE
   */
  SwsContext* retval = sws_getContext(
      ctxt->width, ctxt->height, source_pixel_format, 
      ctxt->width, ctxt->height, dest_pixel_format, 
      SWS_BICUBIC, 0, 0, 0);

  if (!retval) {
    boost::format m("ffmpeg::sws_getContext(src_width=%d, src_height=%d, src_pix_format=`%s', dest_width=%d, dest_height=%d, dest_pix_format=`%s', flags=SWS_BICUBIC, 0, 0, 0) failed: cannot get software scaler context to start encoding or decoding video file `%s'");
    m % ctxt->width % ctxt->height % 
#if LIBAVUTIL_VERSION_INT >= 0x320f01 //50.15.1 @ ffmpeg-0.6
	av_get_pix_fmt_name(source_pixel_format)
#else
	avcodec_get_pix_fmt_name(source_pixel_format)
#endif
      % ctxt->width % ctxt->height % 
#if LIBAVUTIL_VERSION_INT >= 0x320f01 //50.15.1 @ ffmpeg-0.6
	av_get_pix_fmt_name(dest_pixel_format)
#else
	avcodec_get_pix_fmt_name(dest_pixel_format)
#endif
      % filename;
    throw std::runtime_error(m.str());
  }
  return boost::shared_ptr<SwsContext>(retval, std::ptr_fun(deallocate_swscaler));
}

static void deallocate_buffer(uint8_t* p) {
  if (p) av_free(p);
}

boost::shared_array<uint8_t> ffmpeg::make_buffer
(boost::shared_ptr<AVFormatContext> format_context, size_t size) {

  uint8_t* retval=0;

#if LIBAVFORMAT_VERSION_INT < 0x360664 // 54.6.100 @ ffmpeg-0.11

  if (!(format_context->oformat->flags & AVFMT_RAWPICTURE)) {
    /* allocate output buffer */
    /* XXX: API change will be done */
    /* buffers passed into lav* can be allocated any way you prefer,
       as long as they're aligned enough for the architecture, and
       they're freed appropriately (such as using av_free for buffers
       allocated with av_malloc) */
    retval = reinterpret_cast<uint8_t*>(av_malloc(size));

    if (!retval) {
      boost::format m("ffmpeg::av_malloc(%d) failed: could not allocate video output buffer for encoding");
      m % size;
      throw std::runtime_error(m.str());
    }
  }

#endif

  return boost::shared_array<uint8_t>(retval, std::ptr_fun(deallocate_buffer));
}

/**
 * Transforms from Bob's planar 8-bit RGB representation to whatever is
 * required by the FFmpeg encoder output context (peeked from the AVStream
 * object passed).
 */
static void image_to_context(const blitz::Array<uint8_t,3>& data,
    boost::shared_ptr<AVStream> stream,
    boost::shared_ptr<SwsContext> scaler,
    boost::shared_ptr<AVFrame> output_frame) {

  int width = stream->codec->width;
  int height = stream->codec->height;
  
  const uint8_t* datap = data.data();
  int plane_size = width * height;
  const uint8_t* planes[] = {datap+plane_size, datap+2*plane_size, datap, 0};
  int linesize[] = {width, width, width, 0};

#if LIBSWSCALE_VERSION_INT >= 0x000b00 /* 0.11.0 @ ffmpeg-0.6 */
  int ok = sws_scale(scaler.get(), planes, linesize, 0, height, output_frame->data, output_frame->linesize);
#else
  int ok = sws_scale(scaler.get(), const_cast<uint8_t**>(planes), linesize, 0, height, output_frame->data, output_frame->linesize);
#endif
  if (ok < 0) {
    boost::format m("ffmpeg::sws_scale() failed: could not scale frame while encoding - ffmpeg reports error %d");
    m % ok;
    throw std::runtime_error(m.str());
  }
}

/**
 * Transforms from Bob's planar 8-bit RGB representation to whatever is
 * required by the FFmpeg encoder output context (peeked from the AVStream
 * object passed).
 */
static void image_to_context(const blitz::Array<uint8_t,3>& data,
    boost::shared_ptr<AVStream> stream,
    boost::shared_ptr<SwsContext> scaler,
    boost::shared_ptr<AVFrame> output_frame,
    boost::shared_ptr<AVFrame> tmp_frame) {

  int width = stream->codec->width;
  int height = stream->codec->height;
  
  // replace data in the buffer frame by the pixmap to encode
  tmp_frame->linesize[0] = width*3;
  blitz::Array<uint8_t,3> ordered(tmp_frame->data[0],
		  blitz::shape(height, width, 3), blitz::neverDeleteData);
  ordered = const_cast<blitz::Array<uint8_t,3>&>(data).transpose(1,2,0); //organize for ffmpeg

  int ok = sws_scale(scaler.get(), tmp_frame->data, tmp_frame->linesize,
		  0, height, output_frame->data, output_frame->linesize);
  if (ok < 0) {
    boost::format m("ffmpeg::sws_scale() failed: could not scale frame while encoding - ffmpeg reports error %d");
    m % ok;
    throw std::runtime_error(m.str());
  }
}

static void deallocate_codec_context(AVCodecContext* c) {
  int ok = avcodec_close(c);
  if (ok < 0) {
    bob::core::warn << "ffmpeg::avcodec_close() failed: cannot close codec context to stop reading or writing video file (ffmpeg error " << ok << ")" << std::endl;
  }
}

boost::shared_ptr<AVCodecContext> ffmpeg::make_codec_context(
    const std::string& filename, AVStream* stream, AVCodec* codec) {

  AVCodecContext* retval = stream->codec;

  // Hack to correct frame rates that seem to be generated by some codecs
  if(retval->time_base.num > 1000 && retval->time_base.den == 1) {
    retval->time_base.den = 1000;
  }

# if LIBAVCODEC_VERSION_INT < 0x347a00 //52.122.0 @ ffmpeg-0.7

  int ok = avcodec_open(retval, codec);
  if (ok < 0) {
    boost::format m("ffmpeg::avcodec_open(codec=`%s'(0x%x) == `%s') failed: cannot open codec context to start reading or writing video file `%s' - ffmpeg reports error %d == `%s'");
    m % codec->name % codec->id % codec->long_name % filename
      % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

# else //fmpeg >= 0.7

  int ok = avcodec_open2(retval, codec, 0);
  if (ok < 0) {
    boost::format m("ffmpeg::avcodec_open2(codec=`%s'(0x%x) == `%s') failed: cannot open codec context to start reading or writing video file `%s' - ffmpeg reports error %d == `%s'");
    m % codec->name % codec->id % codec->long_name % filename
      % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

# endif

  return boost::shared_ptr<AVCodecContext>(retval, 
      std::ptr_fun(deallocate_codec_context));
}

void ffmpeg::open_output_file(const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context) {

# if LIBAVFORMAT_VERSION_INT < 0x346e00
  // sets the output parameters (must be done even if no parameters).
  int ok = av_set_parameters(format_context.get(), 0);
  if (ok < 0) {
    boost::format m("ffmpeg::av_set_parameters() failed while opening file `%s' for writing: ffmpeg returned error code %d: %s");
    m % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }
  dump_format(format_context.get(), 0, filename.c_str(), 1);
# else
  av_dump_format(format_context.get(), 0, filename.c_str(), 1);
# endif

  /* open the output file, if needed */
  if (!(format_context->oformat->flags & AVFMT_NOFILE)) {
#   if LIBAVFORMAT_VERSION_INT >= 0x346e00 && LIBAVFORMAT_VERSION_INT < 0x350400
    if (avio_open(&format_context->pb, filename.c_str(), URL_WRONLY) < 0) 
#   elif LIBAVFORMAT_VERSION_INT >= 0x350400
    if (avio_open(&format_context->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0) 
#   else
    if (url_fopen(&format_context->pb, filename.c_str(), URL_WRONLY) < 0) 
#   endif
    {
      boost::format m("ffmpeg::avio_open(filename=`%s', AVIO_FLAG_WRITE) failed: cannot open output file for writing");
      m % filename.c_str();
      throw std::runtime_error(m.str());
    }
  }

  /* Write the stream header, if any. */
# if LIBAVFORMAT_VERSION_INT >= 0x346e00 //52.110.0 @ ffmpeg-0.7
  int error = avformat_write_header(format_context.get(), 0);
# else
  int error = av_write_header(format_context.get());
# endif
  if (error < 0) {
    boost::format m("ffmpeg::avformat_write_header(filename=`%s') failed: cannot write header to output file for some reason - ffmpeg reports error %d == `%s'");
    m % filename.c_str() % error % ffmpeg_error(error);
    throw std::runtime_error(m.str());
  }
}

void ffmpeg::close_output_file(const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context) {

  /* Write the trailer, if any. The trailer must be written before you
   * close the CodecContexts open when you wrote the header; otherwise
   * av_write_trailer() may try to use memory that was freed on
   * av_codec_close(). */
  int error = av_write_trailer(format_context.get());
  if (error < 0) {
    boost::format m("ffmpeg::av_write_trailer(filename=`%s') failed: cannot write trailer to output file for some reason - ffmpeg reports error %d == `%s')");
    m % filename % error % ffmpeg_error(error);
    throw std::runtime_error(m.str());
  }

  /* Closes the output file */
# if LIBAVFORMAT_VERSION_INT >= 0x346e00 //52.110.0 @ ffmpeg-0.7
  avio_close(format_context->pb);
# else
  url_fclose(format_context->pb);
# endif

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

void ffmpeg::flush_encoder (const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVStream> stream, AVCodec* codec,
    boost::shared_array<uint8_t> buffer,
    size_t buffer_size) {

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
      ok = av_interleaved_write_frame(format_context.get(), pkt.get());
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
     * The main difference in this way of encoding is what we are responsible
     * for allocating ourselves the buffer in which the encoding will occur. If
     * that space is too small, encoding the data will result in an error. 
     */

    /* encode the image */
    int out_size = avcodec_encode_video(stream->codec, 
        buffer.get(), buffer_size, 0);

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
      pkt.data         = buffer.get();
      pkt.size         = out_size;

      /* Write the compressed frame to the media file. */
      int ok = av_interleaved_write_frame(format_context.get(), &pkt);
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

void ffmpeg::write_video_frame (const blitz::Array<uint8_t,3>& data,
    const std::string& filename,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVStream> stream,
    boost::shared_ptr<AVFrame> context_frame,
    boost::shared_ptr<AVFrame> tmp_frame,
    boost::shared_ptr<SwsContext> swscaler,
    boost::shared_array<uint8_t> buffer,
    size_t buffer_size) {

  if (tmp_frame) 
    image_to_context(data, stream, swscaler, context_frame, tmp_frame);
  else 
    image_to_context(data, stream, swscaler, context_frame);

  if (format_context->oformat->flags & AVFMT_RAWPICTURE) {
    
    /* Raw video case - directly store the picture in the packet */
    AVPacket pkt;
    av_init_packet(&pkt);

    pkt.flags        |= AV_PKT_FLAG_KEY;
    pkt.stream_index  = stream->index;
    pkt.data          = context_frame->data[0];
    pkt.size          = sizeof(AVPicture);

    int ok = av_interleaved_write_frame(format_context.get(), &pkt);
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
    int ok = avcodec_encode_video2(stream->codec, pkt.get(), context_frame.get(), &got_output);
    if (ok < 0) {
      boost::format m("ffmpeg::avcodec_encode_video2() failed: failed to encode video frame while writing to file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }

    /* If size is zero, it means the image was buffered. */
    if (!ok && got_output && pkt->size) {
      if (stream->codec->coded_frame->key_frame) pkt->flags |= AV_PKT_FLAG_KEY;
      pkt->stream_index = stream->index;

      /* Write the compressed frame to the media file. */
      ok = av_interleaved_write_frame(format_context.get(), pkt.get());
      if (ok != 0) {
        boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to write video frame while encoding file `%s' - ffmpeg reports error %d == `%s'");
        m % filename % ok % ffmpeg_error(ok);
        throw std::runtime_error(m.str());
      }

    }

    context_frame->pts += av_rescale_q(1, stream->codec->time_base, 
        stream->time_base);

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
        buffer.get(), buffer_size, context_frame.get());

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
      pkt.data         = buffer.get();
      pkt.size         = out_size;

      /* Write the compressed frame to the media file. */
      int ok = av_interleaved_write_frame(format_context.get(), &pkt);
      av_free_packet(&pkt);
      if (ok && (ok != AVERROR(EINVAL))) {
        boost::format m("ffmpeg::av_interleaved_write_frame() failed: failed to write video frame while encoding file `%s' - ffmpeg reports error %d == `%s'");
        m % filename % ok % ffmpeg_error(ok);
        throw std::runtime_error(m.str());
      }

    }
      
    context_frame->pts += 1;

  }

#endif // FFmpeg version >= 0.11.0
}

static int decode_frame (const std::string& filename, int current_frame,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<SwsContext> scaler,
    boost::shared_ptr<AVFrame> context_frame, uint8_t* data,
    boost::shared_ptr<AVPacket> pkt, 
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

  int ok = avcodec_decode_video2(codec_context.get(), context_frame.get(),
      &got_frame, pkt.get());

#else

  int ok = avcodec_decode_video(codec_context.get(), context_frame.get(), &got_frame, pkt->data, pkt->size);

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

    int conv_height = sws_scale(scaler.get(), context_frame->data,
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

bool ffmpeg::read_video_frame (const std::string& filename, 
    int current_frame, int stream_index,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<SwsContext> swscaler,
    boost::shared_ptr<AVFrame> context_frame, uint8_t* data,
    bool throw_on_error) {

  boost::shared_ptr<AVPacket> pkt = make_packet();

  int ok = 0;
  int got_frame = 0;

  while ((ok = av_read_frame(format_context.get(), pkt.get())) >= 0) {
    if (pkt->stream_index == stream_index) {
      decode_frame(filename, current_frame, codec_context,
          swscaler, context_frame, data, pkt, got_frame,
          throw_on_error);
    }
    av_free_packet(pkt.get());
    if (got_frame) return true; //break loop
  }

#if LIBAVCODEC_VERSION_INT >= 0x344802 //52.72.2 @ ffmpeg-0.6
  if (ok < 0 && ok != (int)AVERROR_EOF) {
    if (throw_on_error) {
      boost::format m("ffmpeg::av_read_frame() failed: on file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
    else return false;
  }
#endif

  // it is the end of the file
  pkt->data = NULL;
  pkt->size = 0;
  //N.B.: got_frame == 0
  const unsigned int MAX_FLUSH_ITERATIONS = 128;
  unsigned int iteration_counter = MAX_FLUSH_ITERATIONS;
  do {
    if (pkt->stream_index == stream_index) {
      decode_frame(filename, current_frame, codec_context,
          swscaler, context_frame, data, pkt, got_frame,
          throw_on_error);
      --iteration_counter;
      if (iteration_counter == 0) {
        if (throw_on_error) {
          boost::format m("ffmpeg::decode_frame() failed: on file `%s' - I've been iterating for over %d times and I cannot find a new frame: this codec (%s) must be buggy!");
          m % filename % MAX_FLUSH_ITERATIONS % codec_context->codec->name; 
          throw std::runtime_error(m.str());
        }
        break;
      }
    }
    else break;
  } while (got_frame == 0);

  return true;
}

static int dummy_decode_frame (const std::string& filename, int current_frame,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<AVFrame> context_frame,
    boost::shared_ptr<AVPacket> pkt, 
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

  int ok = avcodec_decode_video2(codec_context.get(), context_frame.get(),
      &got_frame, pkt.get());

#else

  int ok = avcodec_decode_video(codec_context.get(), context_frame.get(), &got_frame, pkt->data, pkt->size);

#endif

  if (ok < 0 && throw_on_error) {
    boost::format m("ffmpeg::avcodec_decode_video/2() failed: could not skip frame %d of file `%s' - ffmpeg reports error %d == `%s'");
    m % current_frame % filename % ok % ffmpeg_error(ok);
    throw std::runtime_error(m.str());
  }

  return ok;
}

bool ffmpeg::skip_video_frame (const std::string& filename,
    int current_frame, int stream_index,
    boost::shared_ptr<AVFormatContext> format_context,
    boost::shared_ptr<AVCodecContext> codec_context,
    boost::shared_ptr<AVFrame> context_frame,
    bool throw_on_error) {

  boost::shared_ptr<AVPacket> pkt = make_packet();

  int ok = 0;
  int got_frame = 0;

  while ((ok = av_read_frame(format_context.get(), pkt.get())) >= 0) {
    if (pkt->stream_index == stream_index) {
      dummy_decode_frame(filename, current_frame, codec_context,
          context_frame, pkt, got_frame, throw_on_error);
    }
    av_free_packet(pkt.get());
    if (got_frame) return true; //break loop
  }

#if LIBAVCODEC_VERSION_INT >= 0x344802 //52.72.2 @ ffmpeg-0.6
  if (ok < 0 && ok != (int)AVERROR_EOF) {
    if (throw_on_error) {
      boost::format m("ffmpeg::av_read_frame() failed: on file `%s' - ffmpeg reports error %d == `%s'");
      m % filename % ok % ffmpeg_error(ok);
      throw std::runtime_error(m.str());
    }
    else return false;
  }
#endif

  // it is the end of the file
  pkt->data = NULL;
  pkt->size = 0;
  //N.B.: got_frame == 0
  const unsigned int MAX_FLUSH_ITERATIONS = 128;
  unsigned int iteration_counter = MAX_FLUSH_ITERATIONS;
  do {
    if (pkt->stream_index == stream_index) {
      dummy_decode_frame(filename, current_frame, codec_context,
          context_frame, pkt, got_frame, throw_on_error);
      --iteration_counter;
      if (iteration_counter == 0) {
        if (throw_on_error) {
          boost::format m("ffmpeg::decode_frame() failed: on file `%s' - I've been iterating for over %d times and I cannot find a new frame: this codec (%s) must be buggy!");
          m % filename % MAX_FLUSH_ITERATIONS % codec_context->codec->name; 
          throw std::runtime_error(m.str());
        }
        break;
      }
    }
    else break;
  } while (got_frame == 0);

  return true;
}
