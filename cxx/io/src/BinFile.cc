/**
 * @file cxx/io/src/BinFile.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This class can be used to store and load multiarrays into/from files.
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

#include "core/logging.h"
#include "core/array_type.h"

#include "io/BinFile.h"

namespace io = bob::io;
namespace core = bob::core;
namespace ca = bob::core::array;

io::BinFile::BinFile(const std::string& filename, io::BinFile::openmode flag):
  m_header_init(false),
  m_current_array(0),
  m_n_arrays_written(0),
  m_openmode(flag)
{
  if((flag & io::BinFile::out) && (flag & io::BinFile::in)) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::out | 
        std::ios::binary);
    if(m_stream)
    {
      m_header.read(m_stream);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;

      if (flag & io::BinFile::append) {
        m_stream.seekp(0, std::ios::end);
        m_current_array = m_header.m_n_samples;
      }
    }
  }
  else if(flag & io::BinFile::out) {  
    if(m_stream && (flag & io::BinFile::append)) {
      m_stream.open(filename.c_str(), std::ios::out | std::ios::in);
      m_header.read(m_stream);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;
      m_stream.seekp(0, std::ios::end);
      m_current_array = m_header.m_n_samples;
    }
    else
      m_stream.open(filename.c_str(), std::ios::out | std::ios::binary);
  }
  else if(flag & io::BinFile::in) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::binary);
    if(m_stream) {
      m_header.read(m_stream);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;

      if (flag & io::BinFile::append) {
        core::error << "Cannot append data in read only mode." << std::endl;
        throw core::Exception();
      }
    }
  }
  else
  {
    core::error << "Invalid combination of flags." << std::endl;
    throw core::Exception();
  }
}

io::BinFile::~BinFile() {
  close();
}

void io::BinFile::close() {
  // Rewrite the header and update the number of samples
  m_header.m_n_samples = m_n_arrays_written;
  if(m_openmode & io::BinFile::out) m_header.write(m_stream);

  m_stream.close();
}

void io::BinFile::initHeader(const bob::core::array::ElementType type, 
    size_t ndim, const size_t* shape) {
  // Check that data have not already been written
  if (m_n_arrays_written > 0 ) {
    bob::core::error << "Cannot init the header of an output stream in which data" <<
      " have already been written." << std::endl;
    throw bob::core::Exception();
  }

  // Initialize header
  m_header.m_elem_type = type;
  m_header.m_elem_sizeof = bob::core::array::getElementSize(type);
  m_header.setShape(ndim, shape);
  m_header.write(m_stream);
  m_header_init = true;
}

void io::BinFile::write(const ca::interface& data) {

  /**
   * @warning: Please convert your files to HDF5, this format is
   * deprecated starting on 16.04.2011 - AA
   */
  throw core::DeprecationError("BinFile format deprecated on 16.04.2011, use HDF5");

  if(!m_header_init) {
    //initializes the header
    initHeader(data.type().dtype, data.type().nd, data.type().shape);
  }
  else {
      //checks compatibility with previously written stuff
      if (data.type().nd != m_header.getNDim()) 
        throw DimensionError(data.type().nd, m_header.getNDim());
      const size_t* p_shape = data.type().shape;
      const size_t* h_shape = m_header.getShape();
      for (size_t i=0; i<data.type().nd; ++i)
        if(p_shape[i] != h_shape[i]) throw DimensionError(p_shape[i], h_shape[i]);
  }

  // copy the data into the output stream
  m_stream.write((const char*)data.ptr(), data.type().buffer_size());
  
  // increment m_n_arrays_written and m_current_array
  ++m_current_array;
  if (m_current_array>m_n_arrays_written) ++m_n_arrays_written;
}

void io::BinFile::read (ca::interface& a) {
  if(!m_header_init) throw Uninitialized();

  ca::typeinfo compat(getElementType(), m_header.getNDim(), m_header.getShape());
  
  if(!a.type().is_compatible(compat)) a.set(compat);

  m_stream.read((char*)a.ptr(), a.type().buffer_size());
  ++m_current_array;
}

void io::BinFile::read (size_t index, ca::interface& a) {
  // Check that we are reaching an existing array
  if( index >= m_header.m_n_samples ) {
    throw IndexError(index);
  }

  // Set the stream pointer at the correct position
  size_t old_index = m_current_array;
  m_stream.seekg( m_header.getArrayIndex(index) );
  m_current_array = index;

  // Put the content of the stream in the blitz array.
  try {
    return read(a);
  }
  catch (std::invalid_argument& e) {
    m_stream.seekg( m_header.getArrayIndex(old_index) );
    m_current_array = old_index;
    throw e;
  }
}
