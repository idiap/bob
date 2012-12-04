/**
 * @file bob/io/VideoWriter1.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A class to help you write videos. This code originates from the
 * example program: http://cekirdek.pardus.org.tr/~ismail/ffmpeg-docs/output-example_8c-source.html with a few personal modifications.
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

#ifndef BOB_IO_VIDEOWRITER1_H 
#define BOB_IO_VIDEOWRITER1_H

#include <string>
#include <blitz/array.h>
#include <stdint.h>

#include "bob/core/array.h"


extern "C" {
#include <libavutil/pixfmt.h>
}

struct AVOutputFormat;
struct AVFormatContext;
struct AVStream;
struct AVFrame;
struct SwsContext;

namespace bob { namespace io {

  /**
   * Use objects of this class to create and write video files.
   */
  class VideoWriter {

    public:
    
      /**
       * Default constructor, creates a new output file given the input
       * parameters. The codec to be used will be derived from the filename
       * extension.
       *
       * @param filename The name of the file that will contain the video
       * output. If it exists, it will be truncated.
       * @param height The height of the video
       * @param width The width of the video
       * @param framerate The number of frames per second
       * @param bitrate The estimated bitrate of the output video
       * @param gop Group-of-Pictures (how many non-key frames to have per
       * frame group)
       */
      VideoWriter(const std::string& filename, size_t height, size_t width, 
          float framerate=25.f, float bitrate=1500000.f, size_t gop=12);

      /**
       * Destructor virtualization
       */
      virtual ~VideoWriter();

      /**
       * Closes the current video stream and forces writing the trailer. After
       * this point the video becomes invalid.
       */
      void close();

      /**
       * Access to the filename
       */
      inline const std::string& filename() const { return m_filename; }

      /**
       * Access to the height (number of rows) of the video
       */
      inline size_t height() const { return m_height; }

      /**
       * Access to the width (number of columns) of the video
       */
      inline size_t width() const { return m_width; }

      /**
       * Returns the target bitrate for this encoding
       */
      inline float bitRate() const { return m_bitrate; }

      /**
       * Returns the frame rate to be set in the header
       */
      inline float frameRate() const { return m_framerate; }

      /**
       * Returns the group of pictures around key frames
       */
      inline size_t gop() const { return m_gop; }

      /**
       * Duration of the video stream, in microseconds
       */
      inline uint64_t duration() const { 
        return static_cast<uint64_t>(m_current_frame/m_framerate); 
      }

      /**
       * Returns the current number of frames written
       */
      inline size_t numberOfFrames() const { return m_current_frame; }

      /**
       * Returns the name of the codec used for writing this video clip
       */
      inline const std::string& codecName() const { return m_codecname; }

      /**
       * Returns the long version name of the codec used
       */
      inline const std::string& codecLongName() const {
        return m_codecname_long; 
      }

      /**
       * Returns if the video is currently opened for writing
       */
      inline bool is_opened() const { return m_isopen; }
      
      /**
       * Returns a string containing the format information
       */
      std::string info() const;

      /**
       * Compatibility layer type information
       */ 
      const bob::core::array::typeinfo& video_type() const 
      { return m_typeinfo_video; }

      /**
       * Compatibility layer type information
       */ 
      const bob::core::array::typeinfo& frame_type() const 
      { return m_typeinfo_frame; }

      /**
       * Writes a set of frames to the file. The frame set should be setup as a
       * blitz::Array<> with 4 dimensions organized in this way:
       * (frame-number, RGB color-bands, height, width).
       *
       * \warning At present time we only support arrays that have C-style
       * storages (if you pass reversed arrays or arrays with Fortran-style
       * storage, the result is undefined).
       */
      void append(const blitz::Array<uint8_t,4>& data);
    
      /**
       * Writes a new frame to the file. The frame should be setup as a
       * blitz::Array<> with 3 dimensions organized in this way (RGB
       * color-bands, height, width).
       *
       * \warning At present time we only support arrays that have C-style
       * storages (if you pass reversed arrays or arrays with Fortran-style
       * storage, the result is undefined). 
       */
      void append(const blitz::Array<uint8_t,3>& data);

      /**
       * Writes a set of frames to the file. The frame set should be setup as a
       * bob::core::array::interface organized this way: (frame-number, 
       * RGB color-bands, height, width) or (RGB color-bands, height, width).
       */
      void append(const bob::core::array::interface& data);
    
    private: //not implemented

      VideoWriter(const VideoWriter& other);

      VideoWriter& operator= (const VideoWriter& other);

    private: //ffmpeg methods

      /**
       * Adds a video output stream to the file
       */
      AVStream* add_video_stream();

      /**
       * Allocates a picture buffer
       */
      AVFrame* alloc_picture(enum PixelFormat pix_fmt);

      /**
       * Opens the video file
       */
      void open_video();

      /**
       * Closes the video file
       */
      void close_video();

      /**
       * Writes a single video frame into the video file.
       */
      void write_video_frame(const blitz::Array<uint8_t,3>& data);

    private: //representation

      std::string m_filename;
      bool m_isopen;
      size_t m_height;
      size_t m_width;
      float m_framerate;
      float m_bitrate;
      size_t m_gop;
      std::string m_codecname;
      std::string m_codecname_long;
      bob::core::array::typeinfo m_typeinfo_video;
      bob::core::array::typeinfo m_typeinfo_frame;

      AVOutputFormat* m_oformat_ctxt;
      AVFormatContext* m_format_ctxt;
      AVStream* m_video_stream;

      //writing works, but I don't understand these variables...
      AVFrame *picture, *tmp_picture;
      uint8_t *video_outbuf;
      int video_outbuf_size;
          
      size_t m_current_frame; ///< the current frame to be read
      SwsContext* m_sws_context; ///< the color converter

  };

}}

#endif /* BOB_IO_VIDEOWRITER1_H */
