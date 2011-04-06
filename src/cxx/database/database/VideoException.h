/**
 * @file database/VideoException.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * Some exceptions that might be thrown when reading and writing videos.
 */

#ifndef TORCH_DATABASE_VIDEOEXCEPTION_H 
#define TORCH_DATABASE_VIDEOEXCEPTION_H

#include <string>
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

      std::string m_filename;
      std::string m_issue;
      mutable std::string m_message;
  };

}}

#endif /* TORCH_DATABASE_VIDEOEXCEPTION_H */
