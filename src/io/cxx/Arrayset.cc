/**
 * @file io/cxx/Arrayset.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A bob representation of a list of Arrays
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
#include "bob/io/Arrayset.h"
#include "bob/io/CodecRegistry.h"

namespace io = bob::io;

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

io::Arrayset::Arrayset(const std::string& path, char mode) {
  boost::shared_ptr<io::File> file = io::open(path, "", mode);
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
  if (!m_info.is_compatible(array.type())) {
    boost::format s("input array type (%s) is incompatible with this arrayset of type '%s'");
    s % array.type().str() % m_info.str();
    throw std::invalid_argument(s.str());
  }
  m_data.push_back(array);

}

void io::Arrayset::set (size_t id, const Array& array) {

  if (m_data.size() == 0) 
    throw std::runtime_error("cannot set array in empty arrayset");

  if (!m_info.is_compatible(array.type())) {
    boost::format s("input array type (%s) is incompatible with this arrayset of type '%s'");
    s % array.type().str() % m_info.str();
    throw std::invalid_argument(s.str());
  }

  m_data[id] = array;

}

void io::Arrayset::remove (const size_t id) {
  m_data.erase(m_data.begin() + id);
  if (m_data.size() == 0) m_info.reset();
}

void io::Arrayset::save(const std::string& path) {

  //save data to file.
  boost::shared_ptr<io::File> file = io::open(path, "", 'w');
  std::vector<size_t> order;
  order.reserve(m_data.size());
  for (size_t i=0; i<m_data.size(); ++i)
    order.push_back(file->arrayset_append(*m_data[i].get()));

  //flush contents.
  file.reset(); ///< forces closing of the file.

  //reset internal structures
  file = io::open(path, "", 'a');
  for (size_t i=0; i<order.size(); ++i) m_data[i] = Array(file, order[i]);
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
