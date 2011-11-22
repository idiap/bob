/**
 * @file cxx/io/io/VideoException.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Some exceptions that might be thrown when reading and writing videos.
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
