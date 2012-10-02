/**
 * @file io/cxx/Array.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implementation of the Array class.
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

#include "bob/io/Array.h"
#include "bob/io/CodecRegistry.h"

namespace io = bob::io;
namespace ca = bob::core::array;

io::Array::Array(boost::shared_ptr<File> file, size_t index):
  m_external(file),
  m_index(index)
{
}

io::Array::Array(const Array& other): 
  m_external(other.m_external),
  m_index(other.m_index)
{
}

io::Array::~Array() {
}

io::Array& io::Array::operator= (const io::Array& other) {
  m_external = other.m_external;
  m_index = other.m_index;
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
    
boost::shared_ptr<ca::interface> io::Array::get() const {
  boost::shared_ptr<ca::interface> tmp(new ca::blitz_array(external_type()));
  m_external->read(*tmp, m_index);
  return tmp;
}

void io::Array::save(const std::string& path) {
  boost::shared_ptr<File> f = io::open(path, "", 'w');
  f->write(get());
}
