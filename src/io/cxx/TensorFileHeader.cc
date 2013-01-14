/**
 * @file io/cxx/TensorFileHeader.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * This class defines an header for storing multiarrays into .tensor files.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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
#include "bob/core/logging.h"

#include "bob/io/TensorFileHeader.h"
#include "bob/io/Exception.h"

bob::io::detail::TensorFileHeader::TensorFileHeader()
  : m_tensor_type(bob::io::Char),
    m_type(),
    m_n_samples(0),
    m_tensor_size(0)
{
}

bob::io::detail::TensorFileHeader::~TensorFileHeader() { }

size_t bob::io::detail::TensorFileHeader::getArrayIndex (size_t index) const {
  size_t header_size = 7 * sizeof(int);
  return header_size + index * m_tensor_size;
}

void bob::io::detail::TensorFileHeader::read(std::istream& str) {
  // Start reading at the beginning of the stream
  str.seekg(std::ios_base::beg);

  int val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_tensor_type = (bob::io::TensorType)val;
  m_type.dtype = bob::io::tensorTypeToArrayType(m_tensor_type);

  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_n_samples = (size_t)val;
  
  int nd;
  str.read(reinterpret_cast<char*>(&nd), sizeof(int));

  int shape[BOB_MAX_DIM];
  
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  shape[0] = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  shape[1] = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  shape[2] = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  shape[3] = (size_t)val;

  m_type.set_shape(nd, shape);

  header_ok();
}

void bob::io::detail::TensorFileHeader::write(std::ostream& str) const
{
  // Start writing at the beginning of the stream
  str.seekp(std::ios_base::beg);

  int val;
  val = (int)m_tensor_type;
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_n_samples;
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.nd;
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.shape[0];
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.shape[1];
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.shape[2];
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.shape[3];
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
}

void bob::io::detail::TensorFileHeader::header_ok()
{
  // Check the type
  switch (m_tensor_type)
  {
    // supported tensor types
    case bob::io::Char:
    case bob::io::Short:
    case bob::io::Int:
    case bob::io::Long:
    case bob::io::Float:
    case bob::io::Double:
      break;
    // error
    default:
      throw bob::io::UnsupportedTypeError(bob::core::array::t_unknown);
  }

  // Check the number of samples and dimensions
  if( m_type.nd < 1 || m_type.nd > 4) throw bob::io::DimensionError(m_type.nd,4);

  // OK
  update();
}

void bob::io::detail::TensorFileHeader::update()
{
  size_t base_size = 0;
  switch (m_tensor_type)
  {
    case bob::io::Char:    base_size = sizeof(char); break;
    case bob::io::Short:   base_size = sizeof(short); break;
    case bob::io::Int:     base_size = sizeof(int); break;
    case bob::io::Long:    base_size = sizeof(long); break;
    case bob::io::Float:   base_size = sizeof(float); break;
    case bob::io::Double:  base_size = sizeof(double); break;
    default:
      throw bob::io::UnsupportedTypeError(bob::core::array::t_unknown);
  }

  size_t tsize = 1;
  for(size_t i = 0; i < m_type.nd; ++i) tsize *= m_type.shape[i];

  m_tensor_size = tsize * base_size;
}


bob::io::TensorType bob::io::arrayTypeToTensorType(bob::core::array::ElementType eltype)
{
  switch(eltype)
  {
    case bob::core::array::t_int8:
      return bob::io::Char;
    case bob::core::array::t_int16:
      return bob::io::Short;
    case bob::core::array::t_int32:
      return bob::io::Int;
    case bob::core::array::t_int64:
      return bob::io::Long;
    case bob::core::array::t_float32:
      return bob::io::Float;
    case bob::core::array::t_float64:
      return bob::io::Double;
    default:
      throw bob::io::UnsupportedTypeError(bob::core::array::t_unknown);
  }
}
  
bob::core::array::ElementType bob::io::tensorTypeToArrayType(bob::io::TensorType tensortype)
{
  switch(tensortype)
  {
    case bob::io::Char:
      return bob::core::array::t_int8;
    case bob::io::Short:
      return bob::core::array::t_int16;
    case bob::io::Int:
      return bob::core::array::t_int32;
    case bob::io::Long:
      return bob::core::array::t_int64;
    case bob::io::Float:
      return bob::core::array::t_float32;
    case bob::io::Double:
      return bob::core::array::t_float64;
    default:
      throw bob::io::UnsupportedTypeError(bob::core::array::t_unknown);
  }
}
