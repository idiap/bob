/**
 * @file src/cxx/io/src/TensorFile.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to store and load multiarrays into/from files.
 */

#include "core/logging.h"
#include "core/array_type.h"

#include "io/TensorFile.h"

namespace io = Torch::io;
namespace core = Torch::core;
namespace ca = Torch::core::array;

io::TensorFile::TensorFile(const std::string& filename, 
    io::TensorFile::openmode flag):
  m_header_init(false),
  m_current_array(0),
  m_n_arrays_written(0),
  m_openmode(flag)
{
  if((flag & io::TensorFile::out) && (flag & io::TensorFile::in)) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::out | 
        std::ios::binary);
    if(m_stream)
    {
      m_header.read(m_stream);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;

      if (flag & io::TensorFile::append) {
        m_stream.seekp(0, std::ios::end);
        m_current_array = m_header.m_n_samples;
      }
    }
  }
  else if(flag & io::TensorFile::out) {  
    if(m_stream && (flag & io::TensorFile::append)) {
      m_stream.open(filename.c_str(), std::ios::out | std::ios::in |
          std::ios::binary);
      m_header.read(m_stream);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;
      m_stream.seekp(0, std::ios::end);
      m_current_array = m_header.m_n_samples;
    }
    else
      m_stream.open(filename.c_str(), std::ios::out | std::ios::binary);
  }
  else if(flag & io::TensorFile::in) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::binary);
    if(m_stream) {
      m_header.read(m_stream);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;

      if (flag & io::TensorFile::append) {
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

io::TensorFile::~TensorFile() {
  close();
}

void io::TensorFile::peek(ca::typeinfo& info) const {
  info = m_header.m_type;
}

void io::TensorFile::close() {
  // Rewrite the header and update the number of samples
  m_header.m_n_samples = m_n_arrays_written;
  if(m_openmode & io::TensorFile::out) m_header.write(m_stream);

  m_stream.close();
}

void io::TensorFile::initHeader(const ca::typeinfo& info) {
  // Check that data have not already been written
  if (m_n_arrays_written > 0 ) {
    Torch::core::error << "Cannot init the header of an output stream in which data" <<
      " have already been written." << std::endl;
    throw Torch::core::Exception();
  }

  // Initialize header
  m_header.m_type = info;
  m_header.m_tensor_type = io::arrayTypeToTensorType(info.dtype);
  m_header.write(m_stream);
  m_header_init = true;
}

void io::TensorFile::write(const ca::interface& data) {

  const ca::typeinfo& info = data.type();

  if (!m_header_init) initHeader(info);
  else {
    //checks compatibility with previously written stuff
    if (!m_header.m_type.is_compatible(info))
      throw std::runtime_error("buffer does not conform to expected type");
  }
          
  m_stream.write(static_cast<const char*>(data.ptr()), info.buffer_size());

  // increment m_n_arrays_written and m_current_array
  ++m_current_array;
  if (m_current_array>m_n_arrays_written) ++m_n_arrays_written;
}

void io::TensorFile::read (ca::interface& buf) {
  
  if(!m_header_init) throw Uninitialized();
  if(!buf.type().is_compatible(m_header.m_type)) buf.set(m_header.m_type);

  m_stream.read(reinterpret_cast<char*>(buf.ptr()), 
      m_header.m_type.buffer_size());
  ++m_current_array;
}

void io::TensorFile::read (size_t index, ca::interface& buf) {
  
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
