/**
 * @file io/src/Array.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implementation of the Array class.
 */

#include "io/Array.h"

namespace io = Torch::io;

io::Array::Array(const io::buffer& data):
  m_inlined(new io::carray(data))
{
}

io::Array::Array(boost::shared_ptr<buffer> data):
  m_inlined(data)
{
}

io::Array::Array(const std::string& filename, const std::string& codec):
  m_external(new io::filearray(filename, codec))
{
}

io::Array::Array(const Array& other): 
  m_inlined(other.m_inlined),
  m_external(other.m_external)
{
}

io::Array::~Array() {
}

io::Array& io::Array::operator= (const io::Array& other) {
  m_inlined = other.m_inlined;
  m_external = other.m_external;
  return *this;
}

void io::Array::save(const std::string& filename, const std::string& codecname) 
{
  if (m_inlined) {
    m_external.reset(new io::filearray(filename, codecname, true));
    m_external->save(*m_inlined);
    m_inlined.reset();
    return;
  }
  m_external->move(filename, codecname); 
}

const std::string& io::Array::getFilename() const {
  if (m_external) return m_external->info().filename;
  static std::string empty_string;
  return empty_string;
}

boost::shared_ptr<const io::ArrayCodec> io::Array::getCodec() const {
  if (m_external) return m_external->info().codec;
  return boost::shared_ptr<ArrayCodec>(); 
}
    
void io::Array::set(const io::buffer& data) {
  if (m_external) m_external->save(data);
  else m_inlined->set(boost::make_shared<carray>(data));
}
        
void io::Array::set(boost::shared_ptr<buffer> data) {
  if (m_external) m_external->save(*data);
  else m_inlined = data; 
}

boost::shared_ptr<io::buffer> io::Array::get() const {
  if (!m_inlined) {
    boost::shared_ptr<io::buffer> tmp(new carray(m_external->type()));
    m_external->load(*tmp);
    return tmp;
  }
  return m_inlined;
}

void io::Array::load() {
  if (!m_inlined) {
    boost::shared_ptr<io::buffer> tmp(new carray(m_external->type()));
    m_external->load(*tmp);
    m_inlined = tmp;
    m_external.reset();
  }
}
