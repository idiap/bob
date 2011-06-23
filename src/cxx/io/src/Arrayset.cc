/**
 * @file src/cxx/io/src/Arrayset.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch representation of a list of Arrays
 */

#include "io/Arrayset.h"

namespace io = Torch::io;

io::Arrayset::Arrayset ():
  m_inlined(new io::detail::InlinedArraysetImpl()),
  m_external()
{
}

io::Arrayset::Arrayset (const io::detail::InlinedArraysetImpl& inlined):
  m_inlined(new io::detail::InlinedArraysetImpl(inlined)),
  m_external()
{
}

io::Arrayset::Arrayset(const std::string& filename, const std::string& codec):
  m_inlined(),
  m_external(new io::detail::ExternalArraysetImpl(filename, codec))
{
}

io::Arrayset::Arrayset(const io::Arrayset& other):
  m_inlined(other.m_inlined),
  m_external(other.m_external)
{
}

io::Arrayset::~Arrayset() {
}

io::Arrayset& io::Arrayset::operator= (const io::Arrayset& other) {
  m_inlined = other.m_inlined;
  m_external = other.m_external;
  return *this;
}

size_t io::Arrayset::add (boost::shared_ptr<const io::Array> array) {
  return add(*array.get());
}

size_t io::Arrayset::add (const io::Array& array) {
  if (m_inlined) return m_inlined->add(array);
  else return m_external->add(array);
}

size_t io::Arrayset::add (const io::detail::InlinedArrayImpl& array) {
  if (m_inlined) return m_inlined->add(array);
  else return m_external->add(array);
}

size_t io::Arrayset::add (const std::string& filename, const std::string& codec) {
  if (m_inlined) return m_inlined->add(io::Array(filename, codec));
  else return m_external->add(io::Array(filename, codec));
}

void io::Arrayset::set (size_t id, boost::shared_ptr<const io::Array> array) {
  set(id, *array.get());
}

void io::Arrayset::set (size_t id, const io::Array& array) {
  if (m_inlined) (*m_inlined)[id] = array;
  else m_external->set(id, array);
}

void io::Arrayset::set (size_t id, const io::detail::InlinedArrayImpl& array) {
  if (m_inlined) (*m_inlined)[id] = array;
  else m_external->set(id, array);
}

void io::Arrayset::set (size_t id, const std::string& filename, const std::string& codec) {
  if (m_inlined) (*m_inlined)[id] = io::Array(filename, codec);
  else m_external->set(id, io::Array(filename, codec));
}

void io::Arrayset::remove (const size_t id) {
  if (m_inlined) m_inlined->remove(id);
  else m_external->remove(id);
}

Torch::core::array::ElementType io::Arrayset::getElementType() const {
  if (m_inlined) return m_inlined->getElementType();
  return m_external->getElementType();
}

size_t io::Arrayset::getNDim() const {
  if (m_inlined) return m_inlined->getNDim();
  return m_external->getNDim();
}

const size_t* io::Arrayset::getShape() const {
  if (m_inlined) return m_inlined->getShape();
  return m_external->getShape();
}

size_t io::Arrayset::size() const {
  if (m_inlined) return m_inlined->size();
  return m_external->size();
}

void io::Arrayset::save(const std::string& filename, const std::string& codecname) {
  if (m_inlined) {
    m_external.reset(new io::detail::ExternalArraysetImpl(filename, codecname, true));
    m_external->set(*m_inlined);
    m_inlined.reset();
    return;
  }
  m_external->move(filename, codecname); 
}

const std::string& io::Arrayset::getFilename() const {
  if (m_external) return m_external->getFilename();
  static std::string empty_string;
  return empty_string;
}

boost::shared_ptr<const io::ArraysetCodec> io::Arrayset::getCodec() const {
  if (m_external) return m_external->getCodec();
  return boost::shared_ptr<ArraysetCodec>(); 
}
    
void io::Arrayset::load() {
  if (!m_inlined) {
    m_inlined.reset(new detail::InlinedArraysetImpl(m_external->get()));
    m_external.reset();
  }
}

const io::Array io::Arrayset::operator[] (size_t id) const {
  if (m_inlined) return (*m_inlined)[id];
  return (*m_external)[id];
}

io::Array io::Arrayset::operator[] (size_t id) {
  if (m_inlined) return (*m_inlined)[id];
  return (*m_external)[id];
}

io::detail::InlinedArraysetImpl io::Arrayset::get() const {
  if (!m_inlined) return m_external->get();
  return *m_inlined.get();
}
