/**
 * @file io/cxx/TensorFile.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This class can be used to store and load multiarrays into/from files.
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

#include <bob/core/logging.h>
#include <bob/core/array_type.h>

#include <bob/io/TensorFile.h>
#include <bob/io/reorder.h>

bob::io::TensorFile::TensorFile(const std::string& filename, 
    bob::io::TensorFile::openmode flag):
  m_header_init(false),
  m_current_array(0),
  m_n_arrays_written(0),
  m_openmode(flag)
{
  if((flag & bob::io::TensorFile::out) && (flag & bob::io::TensorFile::in)) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::out | 
        std::ios::binary);
    if(m_stream)
    {
      m_header.read(m_stream);
      m_buffer.reset(new char[m_header.m_type.buffer_size()]);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;

      if (flag & bob::io::TensorFile::append) {
        m_stream.seekp(0, std::ios::end);
        m_current_array = m_header.m_n_samples;
      }
    }
  }
  else if(flag & bob::io::TensorFile::out) {  
    if(m_stream && (flag & bob::io::TensorFile::append)) {
      m_stream.open(filename.c_str(), std::ios::out | std::ios::in |
          std::ios::binary);
      m_header.read(m_stream);
      m_buffer.reset(new char[m_header.m_type.buffer_size()]);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;
      m_stream.seekp(0, std::ios::end);
      m_current_array = m_header.m_n_samples;
    }
    else
      m_stream.open(filename.c_str(), std::ios::out | std::ios::binary);
  }
  else if(flag & bob::io::TensorFile::in) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::binary);
    if(m_stream) {
      m_header.read(m_stream);
      m_buffer.reset(new char[m_header.m_type.buffer_size()]);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;

      if (flag & bob::io::TensorFile::append) {
        bob::core::error << "Cannot append data in read only mode." << std::endl;
        throw bob::core::Exception();
      }
    }
  }
  else
  {
    bob::core::error << "Invalid combination of flags." << std::endl;
    throw bob::core::Exception();
  }
}

bob::io::TensorFile::~TensorFile() {
  close();
}

void bob::io::TensorFile::peek(bob::core::array::typeinfo& info) const {
  info = m_header.m_type;
}

void bob::io::TensorFile::close() {
  // Rewrite the header and update the number of samples
  m_header.m_n_samples = m_n_arrays_written;
  if(m_openmode & bob::io::TensorFile::out) m_header.write(m_stream);

  m_stream.close();
}

void bob::io::TensorFile::initHeader(const bob::core::array::typeinfo& info) {
  // Check that data have not already been written
  if (m_n_arrays_written > 0 ) {
    bob::core::error << "Cannot init the header of an output stream in which data have already been written." << std::endl;
    throw bob::core::Exception();
  }

  // Initialize header
  m_header.m_type = info;
  m_header.m_tensor_type = bob::io::arrayTypeToTensorType(info.dtype);
  m_header.write(m_stream);

  // Temporary buffer to help with data transposition...
  m_buffer.reset(new char[m_header.m_type.buffer_size()]);
  
  m_header_init = true;
}

void bob::io::TensorFile::write(const bob::core::array::interface& data) {

  const bob::core::array::typeinfo& info = data.type();

  if (!m_header_init) initHeader(info);
  else {
    //checks compatibility with previously written stuff
    if (!m_header.m_type.is_compatible(info))
      throw std::runtime_error("buffer does not conform to expected type");
  }

  bob::io::row_to_col_order(data.ptr(), m_buffer.get(), info);
          
  m_stream.write(static_cast<const char*>(m_buffer.get()), info.buffer_size());

  // increment m_n_arrays_written and m_current_array
  ++m_current_array;
  if (m_current_array>m_n_arrays_written) ++m_n_arrays_written;
}

void bob::io::TensorFile::read (bob::core::array::interface& buf) {
  
  if(!m_header_init) throw Uninitialized();
  if(!buf.type().is_compatible(m_header.m_type)) buf.set(m_header.m_type);

  m_stream.read(reinterpret_cast<char*>(m_buffer.get()), 
      m_header.m_type.buffer_size());
  
  bob::io::col_to_row_order(m_buffer.get(), buf.ptr(), m_header.m_type);

  ++m_current_array;
}

void bob::io::TensorFile::read (size_t index, bob::core::array::interface& buf) {
  
  // Check that we are reaching an existing array
  if( index > m_header.m_n_samples ) {
    throw IndexError(index);
  }

  // Set the stream pointer at the correct position
  m_stream.seekg( m_header.getArrayIndex(index) );
  m_current_array = index;

  // Put the content of the stream in the blitz array.
  read(buf);
}
