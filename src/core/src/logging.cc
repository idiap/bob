/**
 * @file src/logging.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements all logging infrastructure.
 */

#include "core/logging.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

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

Torch::core::OutputDevice::~OutputDevice() {}
Torch::core::InputDevice::~InputDevice() {}
    
struct NullOutputDevice: public Torch::core::OutputDevice {
  virtual ~NullOutputDevice() {}
  virtual std::streamsize write(const char*, std::streamsize n) {
    return n;
  }
};

struct StdoutOutputDevice: public Torch::core::OutputDevice {
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

struct StderrOutputDevice: public Torch::core::OutputDevice {
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

struct StdinInputDevice: public Torch::core::InputDevice {
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
  return boost::filesystem::extension(filename) == ".gz";
}

struct FileOutputDevice: public Torch::core::OutputDevice {
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

struct FileInputDevice: public Torch::core::InputDevice {
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

bool Torch::core::debug_level(unsigned int i) {
  const char* value = getenv("TORCH_DEBUG");
  if (!value) return false;
  unsigned long v = strtoul(value, 0, 0);
  if (v < 1 || v > 3) v = 0;
  return (i <= v);
}


Torch::core::AutoOutputDevice::AutoOutputDevice()
: m_device(new NullOutputDevice)
{}

Torch::core::AutoOutputDevice::AutoOutputDevice(const Torch::core::AutoOutputDevice& other)
: m_device(other.m_device)
{}

Torch::core::AutoOutputDevice::AutoOutputDevice(const boost::shared_ptr<Torch::core::OutputDevice>& device)
: m_device(device)
{}

Torch::core::AutoOutputDevice::AutoOutputDevice(const std::string& configuration) 
: m_device()
{
  reset(configuration);
}

Torch::core::AutoOutputDevice::~AutoOutputDevice() {}

void Torch::core::AutoOutputDevice::reset(const std::string& configuration) {
  std::string str(configuration);
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  if (str == "null" || str.size()==0) m_device.reset(new NullOutputDevice);
  else if (str == "stdout") m_device.reset(new StdoutOutputDevice);
  else if (str == "stderr") m_device.reset(new StderrOutputDevice);
  else m_device.reset(new FileOutputDevice(configuration));
}

void Torch::core::AutoOutputDevice::reset(const boost::shared_ptr<Torch::core::OutputDevice>& device) {
  m_device = device;
}

std::streamsize Torch::core::AutoOutputDevice::write(const char* s, std::streamsize n) {
  return m_device->write(s, n);
}

void Torch::core::AutoOutputDevice::close() {
  m_device->close();
}

Torch::core::OutputStream::~OutputStream() {}

Torch::core::AutoInputDevice::AutoInputDevice()
: m_device(new StdinInputDevice)
{}

Torch::core::AutoInputDevice::AutoInputDevice(const Torch::core::AutoInputDevice& other)
: m_device(other.m_device)
{}

Torch::core::AutoInputDevice::AutoInputDevice(const boost::shared_ptr<Torch::core::InputDevice>& device)
: m_device(device)
{}

Torch::core::AutoInputDevice::AutoInputDevice(const std::string& configuration) 
: m_device()
{
  reset(configuration);
}

Torch::core::AutoInputDevice::~AutoInputDevice() {}

void Torch::core::AutoInputDevice::reset(const std::string& configuration) {
  std::string str(configuration);
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  if (str == "stdin" || str.size() == 0) m_device.reset(new StdinInputDevice);
  else m_device.reset(new FileInputDevice(configuration));
}

void Torch::core::AutoInputDevice::reset(const boost::shared_ptr<Torch::core::InputDevice>& device) {
  m_device = device;
}

std::streamsize Torch::core::AutoInputDevice::read(char* s, std::streamsize n) {
  return m_device->read(s, n);
}

void Torch::core::AutoInputDevice::close() {
  m_device->close();
}

Torch::core::InputStream::~InputStream() {}

Torch::core::OutputStream Torch::core::debug("stdout");
Torch::core::OutputStream Torch::core::info("stdout");
Torch::core::OutputStream Torch::core::warn("stderr");
Torch::core::OutputStream Torch::core::error("stderr");

std::string Torch::core::tmpdir() {
  const char* value = getenv("TMPDIR");
  if (value) return value;
  else return "/tmp";
}
