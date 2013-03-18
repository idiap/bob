/**
 * @file bob/io/VideoWriter.h
 * @date Wed 28 Nov 2012 13:52:08 CET
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A class to help you write videos. This code originates from
 * http://ffmpeg.org/doxygen/1.0/, "decoding & encoding example".
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

#ifndef BOB_IO_VIDEOWRITER_H
#define BOB_IO_VIDEOWRITER_H

#include "bob/core/array.h"
#include "bob/io/VideoUtilities.h"

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
       * @param gop Group-of-Pictures (emit one intra frame every `gop' frames
       * at most)
       * @param codec If you must, specify a valid FFmpeg codec name here and
       * that will be used to encode the video stream on the output file.
       * @param format If you must, specify a valid FFmpeg output format name
       * and that will be used to encode the video on the output file. Leave
       * it empty to guess from the filename extension.
       */
      VideoWriter(const std::string& filename, size_t height, size_t width,
          double framerate=25., double bitrate=1500000., size_t gop=12,
          const std::string& codec="", const std::string& format="");

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
      inline double bitRate() const { return m_bitrate; }

      /**
       * Returns the frame rate to be set in the header
       */
      inline double frameRate() const { return m_framerate; }

      /**
       * Returns the group of pictures around key frames
       */
      inline size_t gop() const { return m_gop; }

      /**
       * Duration of the video stream, in seconds
       */
      inline uint64_t duration() const { 
        return static_cast<uint64_t>(m_current_frame/m_framerate); 
      }

      /**
       * Returns the current number of frames written
       */
      inline size_t numberOfFrames() const { return m_current_frame; }

      /**
       * Returns if the video is currently opened for writing
       */
      inline bool is_opened() const { return m_opened; }

      /**
       * Some utility information
       */
      std::string formatName() const { 
        return m_format_context->oformat->name;
      }
      std::string formatLongName() const { 
        return m_format_context->oformat->long_name;
      }
      std::string codecName() const { 
        return m_stream->codec->codec->name; 
      }
      std::string codecLongName() const { 
        return m_stream->codec->codec->long_name; 
      }
      
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

    private: //representation
      
      std::string m_filename; ///< file being written
      bool m_opened; ///< is the file currently opened?
      boost::shared_ptr<AVFormatContext> m_format_context; ///< format context
      AVCodec* m_codec; ///< the codec we will be using
      boost::shared_ptr<AVStream> m_stream; ///< the video stream
      boost::shared_ptr<AVCodecContext> m_codec_context; ///< codec context
      boost::shared_ptr<AVFrame> m_context_frame; ///< output frame data
      boost::shared_ptr<AVFrame> m_rgb24_frame; ///< temporary frame data
      boost::shared_ptr<SwsContext> m_swscaler; ///< software scaler
      boost::shared_array<uint8_t> m_buffer; ///< buffer for ffmpeg < 0.11.0
      size_t m_height;
      size_t m_width;
      double m_framerate;
      double m_bitrate;
      size_t m_gop;
      std::string m_codecname;
      std::string m_formatname;
      bob::core::array::typeinfo m_typeinfo_video;
      bob::core::array::typeinfo m_typeinfo_frame;
      size_t m_current_frame;

  };

}}

#endif /* BOB_IO_VIDEOWRITER_H */
