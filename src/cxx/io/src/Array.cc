/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 25 Oct 10:20:52 2011 CEST
 *
 * @brief Implementation of the Array class.
 */

#include "io/Array.h"
#include "io/CodecRegistry.h"

namespace io = Torch::io;

io::Array::Array(const io::buffer& data):
  m_inlined(new io::carray(data)),
  m_loadsall(false)
{
}

io::Array::Array(boost::shared_ptr<buffer> data):
  m_inlined(data),
  m_loadsall(false)
{
}

io::Array::Array(boost::shared_ptr<File> file):
  m_external(file),
  m_index(0),
  m_loadsall(true)
{
}

io::Array::Array(boost::shared_ptr<File> file, size_t index):
  m_external(file),
  m_index(index),
  m_loadsall(false)
{
}

io::Array::Array(const Array& other): 
  m_inlined(other.m_inlined),
  m_external(other.m_external),
  m_index(other.m_index),
  m_loadsall(other.m_loadsall)
{
}

io::Array::Array(const std::string& path):
  m_external(io::open(path, "", 'a')),
  m_index(0),
  m_loadsall(true)
{
}

io::Array::~Array() {
}

io::Array& io::Array::operator= (const io::Array& other) {
  m_inlined = other.m_inlined;
  m_external = other.m_external;
  m_index = other.m_index;
  m_loadsall = other.m_loadsall;
  return *this;
}

const std::string& io::Array::getFilename() const {
  if (m_external) return m_external->filename();
  static std::string empty_string;
  return empty_string;
}

boost::shared_ptr<const io::File> io::Array::getCodec() const {
  if (m_external) return m_external;
  return boost::shared_ptr<File>(); 
}
    
void io::Array::set(const io::buffer& data) {
  m_external.reset();
  m_inlined = boost::make_shared<carray>(data);
}
        
void io::Array::set(boost::shared_ptr<buffer> data) {
  m_external.reset();
  m_inlined = data; 
}

boost::shared_ptr<io::buffer> io::Array::get() const {
  if (!m_inlined) {
    boost::shared_ptr<io::buffer> tmp(new carray(m_external->type(m_loadsall)));
    if (m_loadsall) m_external->read(*tmp);
    else m_external->read(*tmp, m_index);
    return tmp;
  }
  return m_inlined;
}

void io::Array::load() {
  if (!m_inlined) {
    m_inlined.reset(new carray(m_external->type(m_loadsall)));
    if (m_loadsall) m_external->read(*m_inlined);
    else m_external->read(*m_inlined, m_index);
    m_external.reset();
  }
}

void io::Array::append(boost::shared_ptr<File> file) {
  if (m_external) {
    io::carray tmp(m_external->type(m_loadsall));
    m_external->read(tmp, m_index);
    file->append(tmp);
  }
  else {
    file->append(*m_inlined);
  }
}
        
void io::Array::save(const std::string& path) {
  if (m_external) load();
  boost::shared_ptr<File> f = io::open(path, "", 'w');
  f->write(*m_inlined);
  f.reset(); //flush data on file
  m_external = io::open(path, "", 'a');
  m_index = 0;
  m_loadsall = true;
  m_inlined.reset();
}
