/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Converts data from a video format into a Torch5spro multi-array.
 */

#ifndef TORCH5SPRO_IO_VIDEO_ARRAYCODEC_H 
#define TORCH5SPRO_IO_VIDEO_ARRAYCODEC_H

#include "io/ArrayCodec.h"

namespace Torch { namespace io {

  /**
   * Reads and writes single Arrays to video files
   */
  class VideoArrayCodec : public ArrayCodec {

    public:

      VideoArrayCodec();

      virtual ~VideoArrayCodec();

      /**
       * Returns the element type and the number of dimensions of the stored
       * array.
       */
      virtual void peek(const std::string& filename, 
          Torch::core::array::ElementType& eltype, size_t& ndim,
          size_t* shape) const;

      /**
       * Returns the stored array in a InlinedArrayImpl. Because of a
       * limitation of this API you will not be able to retrieve video
       * information that you may find useful, such as the framerate. If you
       * need access to these variables then please use io::VideoReader
       * directly.
       */
      virtual detail::InlinedArrayImpl load(const std::string& filename) const;

      /**
       * Saves a representation of the given array in the file. Please note
       * that writing a video file requires more parameters than the ones given
       * in this method. So, we have to assume certain parameters which are
       * missing and this is a limiation of this implementation. Videos will be
       * saved using the following parameters:
       *
       * -- bitrate = 1500000 bits per second
       * -- framerate = 25 frames per second
       * -- gop = 12 keyframes per normal frames
       *
       * If that is not what you want, then please use io::VideoWriter
       * directly.
       */
      virtual void save (const std::string& filename, 
          const detail::InlinedArrayImpl& data) const;

      /**
       * Returns the name of this codec
       */
      virtual inline const std::string& name () const { return m_name; }

      /**
       * Returns a list of known extensions this codec can handle. The
       * extensions include the initial ".". So, to cover for avi files, you
       * may return a vector containing ".avi" for example. Case matters, so
       * ".avi" and ".AVI" are different extensions. You are the responsible
       * to cover all possible variations an extension can have.
       */
      virtual inline const std::vector<std::string>& extensions () const {
        return m_extensions; 
      }

    private: //representation

      std::string m_name; ///< my own name
      std::vector<std::string> m_extensions; ///< extensions I can handle

  };

}}

#endif /* TORCH5SPRO_IO_VIDEO_ARRAYCODEC_H */
