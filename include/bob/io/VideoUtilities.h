/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 26 Nov 17:33:19 2012 
 *
 * @brief A simple set of utilities to query ffmpeg
 */

#ifndef BOB_IO_VIDEOUTILITIES_H
#define BOB_IO_VIDEOUTILITIES_H

#include <map>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}


namespace bob { namespace io { namespace detail { namespace ffmpeg {

  /**
   * Returns a list of installed codecs, with their capabilities
   */
  void codecs_installed (std::map<std::string, const AVCodecDescriptor*>& installed);

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
