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
#include <boost/filesystem.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/detail/lock.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

Torch::core::Device::~Device() {}
    
struct NullDevice: public Torch::core::Device {
  virtual ~NullDevice() {}
  virtual std::streamsize write(const char*, std::streamsize n) {
    return n;
  }
};

struct StdoutDevice: public Torch::core::Device {
  virtual ~StdoutDevice() {}
  virtual std::streamsize write(const char* s, std::streamsize n) {
    static boost::mutex mutex;
    boost::detail::thread::scoped_lock<boost::mutex> lock(mutex);
    std::cout.write(s, n);
    return n;
  }
};

struct StderrDevice: public Torch::core::Device {
  virtual ~StderrDevice() {}
  virtual std::streamsize write(const char* s, std::streamsize n) {
    static boost::mutex mutex;
    boost::detail::thread::scoped_lock<boost::mutex> lock(mutex);
    std::cerr.write(s, n);
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

struct FileDevice: public Torch::core::Device {
  FileDevice(const std::string& filename)
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
  virtual ~FileDevice() {}
  virtual std::streamsize write(const char* s, std::streamsize n) {
    boost::detail::thread::scoped_lock<boost::mutex> lock(m_mutex);
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

bool Torch::core::debug_level(unsigned int i) {
  const char* value = getenv("TORCH_DEBUG");
  if (!value) return false;
  unsigned long v = strtoul(value, 0, 0);
  if (v < 1 | v > 3) v = 0;
  return (i <= v);
}


Torch::core::Sink::Sink()
: m_device(new NullDevice)
{}

Torch::core::Sink::Sink(const Torch::core::Sink& other)
: m_device(other.m_device)
{}

Torch::core::Sink::Sink(const boost::shared_ptr<Torch::core::Device>& device)
: m_device(device)
{}

Torch::core::Sink::Sink(const std::string& configuration) 
: m_device()
{
  reset(configuration);
}

Torch::core::Sink::~Sink() {}

void Torch::core::Sink::reset(const std::string& configuration) {
  if (configuration == "null") m_device.reset(new NullDevice);
  else if (configuration == "stdout") m_device.reset(new StdoutDevice);
  else if (configuration == "stderr") m_device.reset(new StderrDevice);
  else m_device.reset(new FileDevice(configuration));
}

void Torch::core::Sink::reset(const boost::shared_ptr<Torch::core::Device>& device) {
  m_device = device;
}

std::streamsize Torch::core::Sink::write(const char* s, std::streamsize n) {
  return m_device->write(s, n);
}

void Torch::core::Sink::close() {
  m_device->close();
}

Torch::core::Stream::~Stream() {}

Torch::core::Stream Torch::core::debug("stdout");
Torch::core::Stream Torch::core::info("stdout");
Torch::core::Stream Torch::core::warn("stderr");
Torch::core::Stream Torch::core::error("stderr");
