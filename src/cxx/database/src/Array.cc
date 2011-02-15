/**
 * @file database/src/Array.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implementation of the Array class.
 */

#include "database/Array.h"

namespace db = Torch::database;

db::Array::Array(const db::detail::InlinedArrayImpl& data) :
  m_inlined(new db::detail::InlinedArrayImpl(data))
{
}

db::Array::Array(const std::string& filename, const std::string& codec) :
  m_external(new db::detail::ExternalArrayImpl(filename, codec))
{
}

db::Array::Array(const Array& other) : 
  m_inlined(other.m_inlined),
  m_external(other.m_external)
{
}

db::Array::~Array() {
}

db::Array& db::Array::operator= (const db::Array& other) {
  m_inlined = other.m_inlined;
  m_external = other.m_external;
  return *this;
}

size_t db::Array::getNDim() const {
  if (m_inlined) return m_inlined->getNDim(); 
  return m_external->getNDim();
}

Torch::core::array::ElementType db::Array::getElementType() const {
  if (m_inlined) return m_inlined->getElementType(); 
  return m_external->getElementType();
}

const size_t* db::Array::getShape() const {
  if (m_inlined) return m_inlined->getShape(); 
  return m_external->getShape();
}

void db::Array::save(const std::string& filename, const std::string& codecname) 
{
  if (m_inlined) {
    m_external.reset(new db::detail::ExternalArrayImpl(filename, codecname));
    m_external->set(*m_inlined);
    m_inlined.reset();
    return;
  }
  m_external->move(filename, codecname); 
}

const std::string& db::Array::getFilename() const {
  if (m_external) return m_external->getFilename();
  static std::string empty_string;
  return empty_string;
}

boost::shared_ptr<const db::ArrayCodec> db::Array::getCodec() const {
  if (m_external) return m_external->getCodec();
  return boost::shared_ptr<ArrayCodec>(); 
}
    
void db::Array::set(const db::detail::InlinedArrayImpl& data) {
  if (m_external) m_external.reset();
  m_inlined.reset(new detail::InlinedArrayImpl(data));
}

db::detail::InlinedArrayImpl db::Array::get() const {
  if (!m_inlined) return m_external->get();
  return *m_inlined.get();
}

void db::Array::load() {
  if (!m_inlined) {
    m_inlined.reset(new detail::InlinedArrayImpl(m_external->get()));
    m_external.reset();
  }
}
