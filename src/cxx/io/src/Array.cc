/**
 * @file io/src/Array.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implementation of the Array class.
 */

#include "io/Array.h"

namespace io = Torch::io;

io::Array::Array(const io::detail::InlinedArrayImpl& data) :
  m_inlined(new io::detail::InlinedArrayImpl(data))
{
}

io::Array::Array(const std::string& filename, const std::string& codec) :
  m_external(new io::detail::ExternalArrayImpl(filename, codec))
{
}

io::Array::Array(const Array& other) : 
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

size_t io::Array::getNDim() const {
  if (m_inlined) return m_inlined->getNDim(); 
  return m_external->getNDim();
}

Torch::core::array::ElementType io::Array::getElementType() const {
  if (m_inlined) return m_inlined->getElementType(); 
  return m_external->getElementType();
}

const size_t* io::Array::getShape() const {
  if (m_inlined) return m_inlined->getShape(); 
  return m_external->getShape();
}

void io::Array::save(const std::string& filename, const std::string& codecname) 
{
  if (m_inlined) {
    m_external.reset(new io::detail::ExternalArrayImpl(filename, codecname, true));
    m_external->set(*m_inlined);
    m_inlined.reset();
    return;
  }
  m_external->move(filename, codecname); 
}

const std::string& io::Array::getFilename() const {
  if (m_external) return m_external->getFilename();
  static std::string empty_string;
  return empty_string;
}

boost::shared_ptr<const io::ArrayCodec> io::Array::getCodec() const {
  if (m_external) return m_external->getCodec();
  return boost::shared_ptr<ArrayCodec>(); 
}
    
void io::Array::set(const io::detail::InlinedArrayImpl& data) {
  if (m_external) m_external.reset();
  m_inlined.reset(new detail::InlinedArrayImpl(data));
}

io::detail::InlinedArrayImpl io::Array::get() const {
  if (!m_inlined) return m_external->get();
  return *m_inlined.get();
}

void io::Array::load() {
  if (!m_inlined) {
    m_inlined.reset(new detail::InlinedArrayImpl(m_external->get()));
    m_external.reset();
  }
}
