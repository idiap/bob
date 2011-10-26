/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 25 Oct 17:40:17 2011 CEST
 *
 * @brief A torch representation of a list of Arrays
 */

#include "io/Arrayset.h"
#include "io/CodecRegistry.h"

namespace io = Torch::io;

io::Arrayset::Arrayset ()
{
}

io::Arrayset::Arrayset (boost::shared_ptr<File> file, size_t begin, size_t end)
{
  if (begin >= file->arrayset_size()) return;
  if (end > file->arrayset_size()) end = file->arrayset_size();
  if (begin >= end) return;
  m_data.reserve(end-begin);
  for (size_t i=begin; i<end; ++i) m_data.push_back(io::Array(file, i));
  m_info = file->arrayset_type();
}

io::Arrayset::Arrayset(const std::string& path) {
  boost::shared_ptr<io::File> file = io::open(path, "", 'a');
  m_data.reserve(file->arrayset_size());
  for (size_t i=0; i<file->arrayset_size(); ++i) m_data.push_back(io::Array(file, i));
  m_info = file->arrayset_type();
}

io::Arrayset::Arrayset(const io::Arrayset& other):
  m_data(other.m_data),
  m_info(other.m_info)
{
}

io::Arrayset::~Arrayset() {
}

io::Arrayset& io::Arrayset::operator= (const io::Arrayset& other) {
  m_data = other.m_data;
  m_info = other.m_info;
  return *this;
}

void io::Arrayset::add (const io::Array& array) {

  if (!m_info.is_valid()) { //first addition
    m_data.push_back(array);
    m_info = array.type();
    return;
  }

  //else, check type and add.
  if (!m_info.is_compatible(array.type()))
    throw std::invalid_argument("array type is incompatible with arrayset");
  m_data.push_back(array);

}

void io::Arrayset::set (size_t id, const Array& array) {

  if (m_data.size() == 0) 
    throw std::runtime_error("cannot set array in empty arrayset");

  if (!m_info.is_compatible(array.type()))
    throw std::invalid_argument("array type is incompatible with arrayset");

  m_data[id] = array;

}

void io::Arrayset::remove (const size_t id) {
  m_data.erase(m_data.begin() + id);
  if (m_data.size() == 0) m_info.reset();
}

void io::Arrayset::save(const std::string& path) {
  boost::shared_ptr<io::File> file = io::open(path, "", 'w');
  for (size_t i=0; i<m_data.size(); ++i) {
    file->arrayset_append(*m_data[i].get());
  }
}

void io::Arrayset::load() {
  for (size_t i=0; i<m_data.size(); ++i) m_data[i].load();
}

const io::Array& io::Arrayset::operator[] (size_t id) const {
  return m_data.at(id);
}

io::Array& io::Arrayset::operator[] (size_t id) {
  return m_data.at(id);
}
