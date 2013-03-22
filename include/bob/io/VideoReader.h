/**
 * @file bob/io/VideoReader.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A class to help you read videos. This code originates from
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

#ifndef BOB_IO_VIDEOREADER_H
#define BOB_IO_VIDEOREADER_H

#include <string>
#include <blitz/array.h>
#include <stdint.h>

#include <bob/core/array.h>
#include <bob/io/VideoUtilities.h>

namespace bob { namespace io {

  /**
   * VideoReader objects can read data from video files. The current
   * implementation uses FFMPEG which is a stable freely available
   * implementation for these tasks. You can read an entire video in memory by
   * using the "load()" method or use video iterators to read frame-by-frame
   * and avoid overloading your machine's memory.
   *
   * The maximum precision FFMPEG will output is a 24-bit (8-bit per band)
   * representation of each pixel (32-bit with transparency when supported by
   * bob, which is not the case presently). So, the input of data using this
   * class uses uint8_t as base element type. Output will be colored using the
   * RGB standard, with each band varying between 0 and 255, with zero meaning
   * pure black and 255, pure white (color).
   */
  class VideoReader {

    public:

      /**
       * Opens a new Video stream for reading. The video will be loaded if the
       * combination of format and codec are known to work and have been
       * tested, otherwise an exception is raised. If you set 'check' to
       * 'false', though, we will ignore this check.
       */
      VideoReader(const std::string& filename, bool check=true);

      /**
       * Opens a new Video stream copying information from another VideoStream
       */
      VideoReader(const VideoReader& other);

      /**
       * Destructor virtualization
       */
      virtual ~VideoReader();

      /**
       * Copy operator
       */
      VideoReader& operator= (const VideoReader& other);

      /**
       * Returns the name of the file I'm reading
       */
      inline const std::string& filename() const { return m_filepath; }

      /**
       * Returns the height of the frames in the first video stream.
       */
      inline size_t height() const { return m_height; }

      /**
       * Returns the width of the frames in the first video stream.
       */
      inline size_t width() const { return m_width; }

      /**
       * Returns the number of frames available in this video stream
       */
      inline size_t numberOfFrames() const { return m_nframes; }

      /**
       * Returns the frame rate of the first video stream, in seconds
       */
      double frameRate() const { return m_framerate; }

      /**
       * Duration of the video stream, in microseconds
       */
      inline uint64_t duration() const { return m_duration; }

      /**
       * Returns the format name
       */
      inline const std::string& formatName() const { return m_formatname; }

      /**
       * Returns the longer name for the format
       */
      inline const std::string& formatLongName() const { 
        return m_formatname_long; 
      }

      /**
       * Returns the codec name
       */
      inline const std::string& codecName() const { return m_codecname; }

      /**
       * Returns the longer name for the codec
       */
      inline const std::string& codecLongName() const { 
        return m_codecname_long; 
      }

      /**
       * Returns a string containing the format information
       */
      inline const std::string& info() const { return m_formatted_info; }

      /**
       * Returns the typing information for this video
       */
      inline const bob::core::array::typeinfo& video_type() const 
      { return m_typeinfo_video; }

      /**
       * Returns the typing information for this video
       */
      inline const bob::core::array::typeinfo& frame_type() const 
      { return m_typeinfo_frame; }

      /**
       * Loads all of the video stream in a blitz array organized in this way:
       * (frames, color-bands, height, width). The 'data' parameter will be
       * resized if required.
       *
       * The flag 'throw_on_error' controls the error reporting behavior when
       * reading. By default it is 'false', which means we **won't** report
       * problems reading this stream. We just silently truncate the file. If
       * you set it to 'true', we will report any errors through exceptions. No
       * matter what you chose here, it is your task to verify the return value
       * of this method matches the number of frames indicated by
       * numberOfFrames().
       *
       * The op
       */
      size_t load(blitz::Array<uint8_t,4>& data,
          bool throw_on_error=false, void (*check)(void)=0) const;

      /**
       * Loads all of the video stream in a buffer. Resizes the buffer if
       * the space and type are not good.
       *
       *
       * The flag 'throw_on_error' controls the error reporting behavior when
       * reading. By default it is 'false', which means we **won't** report
       * problems reading this stream. We just silently truncate the file. If
       * you set it to 'true', we will report any errors through exceptions. No
       * matter what you chose here, it is your task to verify the return value
       * of this method matches the number of frames indicated by
       * numberOfFrames().
       */
      size_t load(bob::core::array::interface& b, 
          bool throw_on_error=false, void (*check)(void)=0) const;

    private: //methods

      /**
       * Opens the previously set up Video stream for the reader
       */
      void open(const std::string& filename, bool check);

    public: //iterators

      /**
       * Iterators to video sequences that allow the user to save the state
       * while going through the frames.
       */
      class const_iterator {

        public: //public API for video iterators

          /**
           * Copy constructor. This will cause trigger a deep copy of the other
           * iterator (including all ffmpeg infrastructure). This is expensive,
           * use it carefully.
           */
          const_iterator(const const_iterator& other);

          /**
           * Destructor virtualization
           */
          virtual ~const_iterator();

          /**
           * Assignment operation. This will cause the current state to be
           * dropped and a deep copy of the other iterator (including all
           * ffmpeg infrastructure) to be made. This is expensive, use it
           * carefully.
           */
          const_iterator& operator= (const const_iterator& other);

          /**
           * These various methods will trigger ffmpeg to seek for a
           * particular frame inside the sequence. If you go to far, we will
           * point to "end". If you try to go beyond the begin of the movie, we
           * will clamp at frame 0.
           */

          /**
           * Prefix operator, advance one frame, return self.
           */
          const_iterator& operator++ ();

          /**
           * Suffix operator, advance one frame, return copy.
           */
          //const_iterator operator++ (int); //too inefficient!

          /**
           * Fast-forward the video readout by N frames, return self. This
           * implementation is slow because of ffmpeg limitations. It has no
           * precise frame lookup function. What we have to do is to read
           * frame-by-frame and stop when you want to.
           */
          const_iterator& operator+= (size_t frames);

          /**
           * Compares two iterators for equality
           */
          bool operator== (const const_iterator& other);

          /**
           * Compares two iterators for inequality
           */
          bool operator!= (const const_iterator& other);
          
          /**
           * Reads the currently pointed frame and advances one position.
           * Please note that when you call this method in a loop, you don't
           * need to increment the iterator as it auto-increments itself. The
           * 'data' format is (color-bands, height, width). If the size does
           * not match the movie specifications, the array data will be
           * resized.
           *
           * Once the end position is reached, the ffmpeg infrastructure is
           * automatically destroyed. Rewinding the iterator will cause a
           * re-load of that infrastructure. If we have reached the end
           * position, an exception is raised if you try to read() the
           * iterator.
           *
           * The flag 'throw_on_error' controls the error reporting behavior
           * when reading. By default it is 'false', which means we **won't**
           * report problems reading this stream. We just silently truncate the
           * file. If you set it to 'true', we will report any errors through
           * exceptions. No matter what you chose here, it is your task to
           * verify the return value of this method matches the number of
           * frames indicated by numberOfFrames().
           */
          bool read (bob::core::array::interface& b, bool throw_on_error=false);

          /**
           * Reads the currently pointed frame and advances one position.
           * Please note that when you call this method in a loop, you don't
           * need to increment the iterator as it auto-increments itself. The
           * 'data' format is (color-bands, height, width). If the size does
           * not match the movie specifications, the array data will be
           * resized.
           *
           * Once the end position is reached, the ffmpeg infrastructure is
           * automatically destroyed. Rewinding the iterator will cause a
           * re-load of that infrastructure. If we have reached the end
           * position, an exception is raised if you try to read() the
           * iterator.
           *
           * The flag 'throw_on_error' controls the error reporting behavior
           * when reading. By default it is 'false', which means we **won't**
           * report problems reading this stream. We just silently truncate the
           * file. If you set it to 'true', we will report any errors through
           * exceptions. No matter what you chose here, it is your task to
           * verify the return value of this method matches the number of
           * frames indicated by numberOfFrames().
           */
          bool read (blitz::Array<uint8_t,3>& data, bool throw_on_error=false);

          /**
           * Resets the current iterator state by closing and re-opening the
           * movie file and positioning the frame pointer to the first frame in
           * the sequence.
           */
          void reset();

          /**
           * Tells the current frame number
           */
          inline size_t cur() const { return m_current_frame; }

          /**
           * Gets the parent
           */
          const VideoReader* parent() const { return m_parent; }

        private: //cannot create or copy iterators
         
          /**
           * The only way to build a new iterator is to use the parent's
           * begin()/end() methods.
           */
          const_iterator(const VideoReader* parent);

          /**
           * This creates an iterator pointing to "end"
           */
          const_iterator();

        private: //methods

          /**
           * Initializes this iterator from scratch
           */
          void init();

        private: //representation
          const VideoReader* m_parent; ///< who generated me
          boost::shared_ptr<AVFormatContext> m_format_context; ///< format context
          int m_stream_index; ///< which stream in the file points to the video
          AVCodec* m_codec; ///< the codec we will be using
          boost::shared_ptr<AVStream> m_stream; ///< the video stream
          boost::shared_ptr<AVCodecContext> m_codec_context; ///< format context
          boost::shared_ptr<AVFrame> m_context_frame; ///< from file
          blitz::Array<uint8_t,3> m_rgb_array; ///< temporary
          boost::shared_ptr<SwsContext> m_swscaler; ///< software scaler
          size_t m_current_frame; ///< the current frame to be read

        public: //friendship

          friend class VideoReader; //required for construction
      };


      /**
       * Returns an iterator to the begin of the video stream.
       */
      const_iterator begin() const;

      /**
       * Returns an iterator to the end of the video sequence.
       */
      const_iterator end() const;

    private: //our representation

      std::string m_filepath; ///< the name of the file we are manipulating
      bool m_check; ///< shall I check for compatibility when opening?
      size_t m_height; ///< the height of the video frames (number of rows)
      size_t m_width; ///< the width of the video frames (number of columns)
      size_t m_nframes; ///< the number of frames in this video file
      double m_framerate; ///< rate of frames in the video stream
      uint64_t m_duration; ///< in microsseconds, for the whole video
      std::string m_formatname; ///< the name of the ffmpeg format to be used
      std::string m_formatname_long; ///< long version of m_formatname
      std::string m_codecname; ///< the name of the ffmpeg codec to be used
      std::string m_codecname_long; ///< long version of m_codecname
      std::string m_formatted_info; ///< printable information about the video
      bob::core::array::typeinfo m_typeinfo_video; ///< read whole video type
      bob::core::array::typeinfo m_typeinfo_frame; ///< read single frame type
  };

}}

#endif //BOB_IO_VIDEOREADER_H
