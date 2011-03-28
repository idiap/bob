/**
 * @file database/VideoException.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * Some exceptions that might be thrown when reading and writing videos.
 */

#ifndef TORCH_DATABASE_VIDEOEXCEPTION_H 
#define TORCH_DATABASE_VIDEOEXCEPTION_H

#include "database/Exception.h"

namespace Torch { namespace database {

  /**
   * Thrown when there is a problem with a Video file
   */
  class FFmpegException: public database::Exception {
    public:
      FFmpegException(const char* filename, const char* issue) throw(); 
      virtual ~FFmpegException() throw();
      virtual const char* what() const throw();

    private:

      const char* m_filename;
      const char* m_issue;
      mutable std::string m_message;
  };

  /**
   * Unsupported operation is thrown when you open a video for writing and try
   * to read something from it or vice-versa.
   */
  class UnsupportedOperation: public database::Exception {
    public:
      UnsupportedOperation(const char* filename, bool read) throw();
      virtual ~UnsupportedOperation() throw();
      virtual const char* what() const throw();

    private:

      const char* m_filename;
      bool m_read;
      mutable std::string m_message;
  };

}}

#endif /* TORCH_DATABASE_VIDEOEXCEPTION_H */
