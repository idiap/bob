/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 26 Mar 18:56:01 2011 
 *
 * @brief Implements video read/write exceptions
 */

#include <boost/format.hpp>
#include "io/VideoException.h"

namespace io = Torch::io;

io::FFmpegException::FFmpegException(const char* filename,
    const char* issue) throw(): 
  m_filename(filename),
  m_issue(issue)
{
}

io::FFmpegException::~FFmpegException() throw() {
}

const char* io::FFmpegException::what() const throw() {
  try {
    boost::format message("%s: %s");
    message % m_filename % m_issue;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::FFmpegException: cannot format, exception raised";
    return emergency;
  }
}
