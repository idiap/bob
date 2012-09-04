/**
 * @file cxx/io/src/VideoException.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements video read/write exceptions
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include <boost/format.hpp>
#include "bob/io/VideoException.h"

namespace io = bob::io;

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

io::VideoIsClosed::VideoIsClosed(const char* filename) throw(): 
  m_filename(filename)
{
}

io::VideoIsClosed::~VideoIsClosed() throw() {
}

const char* io::VideoIsClosed::what() const throw() {
  try {
    boost::format message("Cannot write to %s anymore, video was already closed by user");
    message % m_filename;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::VideoIsClosed: cannot format, exception raised";
    return emergency;
  }
}
