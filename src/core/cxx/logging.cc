/**
 * @file core/cxx/logging.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements all logging infrastructure.
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

#include "bob/core/logging.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/shared_array.hpp>

/**
 * MT "lock" support was only introduced in Boost 1.35. Before copying this
 * very ugly hack, make sure we are still using Boost 1.34. This will no longer
 * be the case starting January 2011.
 */
#include <boost/version.hpp>
#include <boost/thread/mutex.hpp>
#if ((BOOST_VERSION / 100) % 1000) > 34
#include <boost/thread/locks.hpp>
#else
#warning Disabling MT locks because Boost < 1.35!
#endif

bob::core::OutputDevice::~OutputDevice() {}
bob::core::InputDevice::~InputDevice() {}
    
struct NullOutputDevice: public bob::core::OutputDevice {
  virtual ~NullOutputDevice() {}
  virtual std::streamsize write(const char*, std::streamsize n) {
    return n;
  }
};

struct StdoutOutputDevice: public bob::core::OutputDevice {
  virtual ~StdoutOutputDevice() {}
  virtual std::streamsize write(const char* s, std::streamsize n) {
    static boost::mutex mutex;
#if ((BOOST_VERSION / 100) % 1000) > 35
    boost::lock_guard<boost::mutex> lock(mutex);
#endif
    std::cout.write(s, n);
    return n;
  }
};

struct StderrOutputDevice: public bob::core::OutputDevice {
  virtual ~StderrOutputDevice() {}
  virtual std::streamsize write(const char* s, std::streamsize n) {
    static boost::mutex mutex;
#if ((BOOST_VERSION / 100) % 1000) > 35
    boost::lock_guard<boost::mutex> lock(mutex);
#endif
    std::cerr.write(s, n);
    return n;
  }
};

struct StdinInputDevice: public bob::core::InputDevice {
  virtual ~StdinInputDevice() {}
  virtual std::streamsize read(char* s, std::streamsize n) {
    static boost::mutex mutex;
#if ((BOOST_VERSION / 100) % 1000) > 35
    boost::lock_guard<boost::mutex> lock(mutex);
#endif
    std::cin.read(s, n);
    return n;
  }
};

/**
 * Determines if the input filename ends in ".gz"
 *
 * @param filename The name of the file to be analyzed.
 */
inline static bool is_dot_gz(const std::string& filename) {
  return boost::filesystem::path(filename).extension() == ".gz";
}

struct FileOutputDevice: public bob::core::OutputDevice {
  FileOutputDevice(const std::string& filename)
    : m_filename(filename),
      m_file(),
      m_ostream(),
      m_mutex()
  {
    //this first bit creates the output file handle
    std::ios_base::openmode mode = std::ios_base::out | std::ios_base::trunc;
    if (is_dot_gz(filename)) mode |= std::ios_base::binary;
    m_file.open(filename.c_str(), mode);
    //this second part configures gzip'ing if necessary and associates the
    //output file with the filtering stream.
    if (is_dot_gz(filename)) 
      m_ostream.push(boost::iostreams::basic_gzip_compressor<>());
    m_ostream.push(m_file);
  }
  virtual ~FileOutputDevice() {}
  virtual std::streamsize write(const char* s, std::streamsize n) {
#if ((BOOST_VERSION / 100) % 1000) > 35
    boost::lock_guard<boost::mutex> lock(m_mutex);
#endif
    m_ostream.write(s, n);
    return n;
  }

  //internal representation
  private:
    std::string m_filename; ///< the name of the file I'm writing to
    std::ofstream m_file; ///< the file output stream
    boost::iostreams::filtering_ostream m_ostream; ///< the output stream
    boost::mutex m_mutex; ///< multi-threading guardian

};

struct FileInputDevice: public bob::core::InputDevice {
  FileInputDevice(const std::string& filename)
    : m_filename(filename),
      m_file(),
      m_istream(),
      m_mutex()
  {
    //this first bit creates the input file handle
    std::ios_base::openmode mode = std::ios_base::in;
    if (is_dot_gz(filename)) mode |= std::ios_base::binary;
    m_file.open(filename.c_str(), mode);
    //this second part configures gzip'ing if necessary and associates the
    //input file with the filtering stream.
    if (is_dot_gz(filename)) 
      m_istream.push(boost::iostreams::basic_gzip_decompressor<>());
    m_istream.push(m_file);
  }
  virtual ~FileInputDevice() {}
  virtual std::streamsize read(char* s, std::streamsize n) {
#if ((BOOST_VERSION / 100) % 1000) > 35
    boost::lock_guard<boost::mutex> lock(m_mutex);
#endif
    m_istream.read(s, n);
    return n;
  }

  //internal representation
  private:
    std::string m_filename; ///< the name of the file I'm reading from
    std::ifstream m_file; ///< the file input stream
    boost::iostreams::filtering_istream m_istream; ///< the input stream
    boost::mutex m_mutex; ///< multi-threading guardian

};

bool bob::core::debug_level(unsigned int i) {
  const char* value = getenv("BOB_DEBUG");
  if (!value) return false;
  unsigned long v = strtoul(value, 0, 0);
  if (v < 1 || v > 3) v = 0;
  return (i <= v);
}


bob::core::AutoOutputDevice::AutoOutputDevice()
: m_device(new NullOutputDevice)
{}

bob::core::AutoOutputDevice::AutoOutputDevice(const AutoOutputDevice& other)
: m_device(other.m_device)
{}

bob::core::AutoOutputDevice::AutoOutputDevice(const boost::shared_ptr<OutputDevice>& device)
: m_device(device)
{}

bob::core::AutoOutputDevice::AutoOutputDevice(const std::string& configuration) 
: m_device()
{
  reset(configuration);
}

bob::core::AutoOutputDevice::~AutoOutputDevice() {}

void bob::core::AutoOutputDevice::reset(const std::string& configuration) {
  std::string str(configuration);
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  if (str == "null" || str.size()==0) m_device.reset(new NullOutputDevice);
  else if (str == "stdout") m_device.reset(new StdoutOutputDevice);
  else if (str == "stderr") m_device.reset(new StderrOutputDevice);
  else m_device.reset(new FileOutputDevice(configuration));
}

void bob::core::AutoOutputDevice::reset(const boost::shared_ptr<OutputDevice>& device) {
  m_device = device;
}

std::streamsize bob::core::AutoOutputDevice::write(const char* s, std::streamsize n) {
  return m_device->write(s, n);
}

void bob::core::AutoOutputDevice::close() {
  m_device->close();
}

bob::core::OutputStream::~OutputStream() {}

bob::core::AutoInputDevice::AutoInputDevice()
: m_device(new StdinInputDevice)
{}

bob::core::AutoInputDevice::AutoInputDevice(const AutoInputDevice& other)
: m_device(other.m_device)
{}

bob::core::AutoInputDevice::AutoInputDevice(const boost::shared_ptr<InputDevice>& device)
: m_device(device)
{}

bob::core::AutoInputDevice::AutoInputDevice(const std::string& configuration) 
: m_device()
{
  reset(configuration);
}

bob::core::AutoInputDevice::~AutoInputDevice() {}

void bob::core::AutoInputDevice::reset(const std::string& configuration) {
  std::string str(configuration);
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  if (str == "stdin" || str.size() == 0) m_device.reset(new StdinInputDevice);
  else m_device.reset(new FileInputDevice(configuration));
}

void bob::core::AutoInputDevice::reset(const boost::shared_ptr<InputDevice>& device) {
  m_device = device;
}

std::streamsize bob::core::AutoInputDevice::read(char* s, std::streamsize n) {
  return m_device->read(s, n);
}

void bob::core::AutoInputDevice::close() {
  m_device->close();
}

bob::core::InputStream::~InputStream() {}

bob::core::OutputStream bob::core::debug("stdout");
bob::core::OutputStream bob::core::info("stdout");
bob::core::OutputStream bob::core::warn("stderr");
bob::core::OutputStream bob::core::error("stderr");

std::string bob::core::tmpdir() {
  const char* value = getenv("TMPDIR");
  if (value) return value;
  else return "/tmp";
}

std::string bob::core::tmpfile(const std::string& extension) {
  boost::filesystem::path tpl = bob::core::tmpdir();
  tpl /= std::string("bobtest_core_binformatXXXXXX") + extension;
  boost::shared_array<char> char_tpl(new char[tpl.string().size()+1]);
  strcpy(char_tpl.get(), tpl.string().c_str());
  int fd = mkstemps(char_tpl.get(), extension.size());
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
}
