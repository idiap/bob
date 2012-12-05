/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 26 Nov 17:33:19 2012 
 *
 * @brief A simple set of utilities to query ffmpeg
 * ##
 * FFMpeg versions for your reference
 * ffmpeg | avformat   |  avcodec     
 * =======+============+=============
 * 0.5    | 52. 31.  0 | 52. 20.  1  
 * 0.6    | 52. 64.  2 | 52. 72.  2  
 * 0.7    | 52.111.  0 | 52.123.  0  
 * 0.8    | 53.  5.  0 | 53.  8.  0  
 * 0.9    | 53. 24.  2 | 53. 42.  4  
 * 0.10   | 53. 32.100 | 53. 61.100  
 * 0.11   | 54.  6.100 | 54. 23.100  
 * 1.0    | 54. 29.104 | 54. 59.100  
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

#ifndef BOB_IO_VIDEOUTILITIES_H
#define BOB_IO_VIDEOUTILITIES_H

#include <map>
#include <vector>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

/**
 * These macros will ease the handling of ffmpeg versions
 */
#if   LIBAVFORMAT_VERSION_INT < 0x341f00 && LIBAVCODEC_VERSION_INT < 0x341401
#error Bob can only be compiled against FFmpeg >= 0.5.0
#elif LIBAVFORMAT_VERSION_INT < 0x344002 && LIBAVCODEC_VERSION_INT < 0x344802
#define FFMPEG_VERSION_INT 0x000500 //0.5.0
#elif LIBAVFORMAT_VERSION_INT < 0x346f00 && LIBAVCODEC_VERSION_INT < 0x347b00
#define FFMPEG_VERSION_INT 0x000600 //0.6.0
#elif LIBAVFORMAT_VERSION_INT < 0x350500 && LIBAVCODEC_VERSION_INT < 0x350800
#define FFMPEG_VERSION_INT 0x000700 //0.7.0
#elif LIBAVFORMAT_VERSION_INT < 0x351802 && LIBAVCODEC_VERSION_INT < 0x352a04
#define FFMPEG_VERSION_INT 0x000800 //0.8.0
#elif LIBAVFORMAT_VERSION_INT < 0x352064 && LIBAVCODEC_VERSION_INT < 0x353d64
#define FFMPEG_VERSION_INT 0x000900 //0.9.0
#elif LIBAVFORMAT_VERSION_INT < 0x360664 && LIBAVCODEC_VERSION_INT < 0x361764
#define FFMPEG_VERSION_INT 0x000a00 //0.10.0
#elif LIBAVFORMAT_VERSION_INT < 0x361d68 && LIBAVCODEC_VERSION_INT < 0x363b64
#define FFMPEG_VERSION_INT 0x000b00 //0.11.0
#else
#define FFMPEG_VERSION_INT 0x010000 //1.0.0
#endif

/**
 * Some code to account for older versions of ffmpeg
 */
#ifndef AV_CODEC_ID_NONE
#define AV_CODEC_ID_NONE CODEC_ID_NONE
#define AV_CODEC_ID_MPEG1VIDEO CODEC_ID_MPEG1VIDEO
#define AV_CODEC_ID_MPEG2VIDEO CODEC_ID_MPEG2VIDEO
typedef CodecID AVCodecID;
#endif

#ifndef AV_PIX_FMT_RGB24
#define AV_PIX_FMT_RGB24 PIX_FMT_RGB24
#endif

#ifndef AV_PIX_FMT_YUV420P
#define AV_PIX_FMT_YUV420P PIX_FMT_YUV420P
#endif

#ifndef AV_PIX_FMT_NONE
#define AV_PIX_FMT_NONE PIX_FMT_NONE
#endif

#ifndef AV_PKT_FLAG_KEY
#define AV_PKT_FLAG_KEY PKG_FLAG_KEY
#endif

namespace bob { namespace io { namespace detail { namespace ffmpeg {

  /**
   * Breaks a list of words separated by commands into a word list
   */
  void tokenize_csv(const char* what, std::vector<std::string>& values);

  /**
   * Returns a list of installed codecs, with their capabilities
   */
  void codecs_installed (std::map<std::string, const AVCodec*>& installed);

  /**
   * Returns a list of input formats this installation can handle, with details
   */
  void iformats_installed (std::map<std::string, AVInputFormat*>& installed);

  /**
   * Returns a list of output formats this installation can handle, with
   * details
   */
  void oformats_installed (std::map<std::string, AVOutputFormat*>& installed);

}}}}

#endif /* BOB_IO_VIDEOUTILITIES_H */
