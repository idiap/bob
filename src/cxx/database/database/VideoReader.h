/**
 * @file database/VideoReader.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * Implements a class to read Video files and convert the frames into something
 * that torch can understand (i.e. blitz::Array<>'s). This implementation is
 * heavily based on FFmpeg and the excellent tutorial here:
 * http://dranger.com/ffmpeg/, with some personal modifications. In doubt,
 * consult the ffmpeg documentation: http://ffmpeg.org/documentation.html
 */

#ifndef TORCH_DATABASE_DETAIL_VIDEOREADER_H
#define TORCH_DATABASE_DETAIL_VIDEOREADER_H

#include <string>
#include <blitz/array.h>
#include <stdint.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

namespace Torch { namespace database {

  /**
   * VideoReader objects can read data from video files. The current
   * implementation uses FFMPEG which is a stable freely available
   * implementation for these tasks. You can read an entire video in memory by
   * using the "load()" method or use video iterators to read frame-by-frame
   * and avoid overloading your machine's memory.
   *
   * The maximum precision FFMPEG will output is a 24-bit (8-bit per band)
   * representation of each pixel (32-bit with transparency when supported by
   * Torch, which is not the case presently). So, the input of data using this
   * class uses uint8_t as base element type. Output will be colored using the
   * RGB standard, with each band varying between 0 and 255, with zero meaning
   * pure black and 255, pure white (color).
   */
  class VideoReader {

    public:

      /**
       * Opens a new Video stream for reading. 
       */
      VideoReader(const std::string& filename);

      /**
       * Destructor virtualization
       */
      virtual ~VideoReader();

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
      float frameRate() const { return m_framerate; }

      /**
       * Duration of the video stream, in microseconds
       */
      inline uint64_t duration() const { return m_duration; }

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
       * Loads all of the video stream in a blitz array organized in this way:
       * (frames, color-bands, height, width). The 'data' parameter will be
       * resized if required.
       */
      void load(blitz::Array<uint8_t,4>& data) const;

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
           * need to increment the iterator as it auto-increments it. The
           * 'data' format is (color-bands, height, width). If the size does
           * not match the movie specifications, the array data will be
           * resized.
           *
           * Once the end position is reached, the ffmpeg infrastructure is
           * automatically destroyed. Rewinding the iterator will cause a
           * re-load of that infrastructure. If we have reached the end
           * position, an exception is raised if you try to read() the
           * iterator.
           */
          void read (blitz::Array<uint8_t,3>& data);

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
          AVFormatContext* m_format_ctxt; ///< useful, but for what? ;-)
          int m_stream_index; ///< in which stream is the video feed?
          AVCodecContext* m_codec_ctxt; ///< context that contains first video
          AVCodec* m_codec; ///< codec that is used for decoding the video
          AVFrame* m_frame_buffer; ///< ffmpeg native frame format
          AVFrame* m_rgb_frame_buffer; ///< ffmpeg frame converted into RGB
          uint8_t* m_raw_buffer; ///< raw representation of the RGB data
          size_t m_current_frame; ///< the current frame to be read
          SwsContext* m_sws_context; ///< the color converter

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

    private: //static stuff that is required.

      static bool s_ffmpeg_initialized; ///< is ffmpeg initialized?

    private: //our representation

      std::string m_filepath; ///< the name of the file we are manipulating
      size_t m_height; ///< the height of the video frames (number of rows)
      size_t m_width; ///< the width of the video frames (number of columns)
      size_t m_nframes; ///< the number of frames in this video file
      float m_framerate; ///< rate of frames in the video stream
      uint64_t m_duration; ///< in microsseconds, for the whole video
      std::string m_codecname; ///< the name of the ffmpeg codec to be used
      std::string m_codecname_long; ///< long version of m_codecname
      std::string m_formatted_info; ///< printable information about the video
  };

}}

#endif //TORCH_DATABASE_DETAIL_VIDEOREADER_H
