/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 26 Mar 18:56:01 2011 
 *
 * @brief Implements video read/write exceptions
 */

#include <boost/format.hpp>
#include "database/VideoException.h"

namespace db = Torch::database;

db::FFmpegException::FFmpegException(const char* filename,
    const char* issue) throw(): 
  m_filename(filename),
  m_issue(issue)
{
}

db::FFmpegException::~FFmpegException() throw() {
}

const char* db::FFmpegException::what() const throw() {
  try {
    boost::format message("%s: %s");
    message % m_filename % m_issue;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::FFmpegException: cannot format, exception raised";
    return emergency;
  }
}
