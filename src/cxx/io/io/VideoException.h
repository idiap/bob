/**
 * @file io/VideoException.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * Some exceptions that might be thrown when reading and writing videos.
 */

#ifndef TORCH_IO_VIDEOEXCEPTION_H 
#define TORCH_IO_VIDEOEXCEPTION_H

#include <string>
#include "io/Exception.h"

namespace Torch { namespace io {

  /**
   * Thrown when there is a problem with a Video file
   */
  class FFmpegException: public io::Exception {
    public:
      FFmpegException(const char* filename, const char* issue) throw(); 
      virtual ~FFmpegException() throw();
      virtual const char* what() const throw();

    private:

      std::string m_filename;
      std::string m_issue;
      mutable std::string m_message;
  };

  /**
   * Thrown if a writeable video is already closed and the user tries to write
   * on it.
   */
  class VideoIsClosed: public io::Exception {
    public:
      VideoIsClosed(const char* filename) throw();
      virtual ~VideoIsClosed() throw();
      virtual const char* what() const throw();

    private:

      std::string m_filename;
      mutable std::string m_message;
  };

}}

#endif /* TORCH_IO_VIDEOEXCEPTION_H */
