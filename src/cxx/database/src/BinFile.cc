/**
 * @file src/cxx/database/src/BinFile.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to store and load multiarrays into/from files.
 */


#include "database/BinFile.h"

namespace db = Torch::database;

db::BinFile::BinFile(const std::string& filename, db::openmode flag):
  m_header_init(false),
  m_current_array(0),
  m_n_arrays_written(0),
  m_openmode(flag)
{
  if((flag & db::BinFile::out) && (flag & db::BinFile::in)) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::out | 
        std::ios::binary);
    m_header.read(m_stream);
    m_header_init = true;
    m_n_arrays_written = m_header.m_n_samples;

    if (flag & db::BinFile::append) {
      m_stream.seekp(0, std::ios::end);
      m_current_array = m_header.m_n_samples;
    }
  }
  else if(flag & db::BinFile::out) {
    m_stream.open(filename.c_str(), std::ios::out | std::ios::binary);

    if (flag & db::BinFile::append) {
      m_header.read(m_stream);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;
      m_stream.seekp(0, std::ios::end);
      m_current_array = m_header.m_n_samples;

    }
  }
  else if(flag & db::BinFile::in) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::binary);
    m_header.read(m_stream);
    m_header_init = true;
    m_n_arrays_written = m_header.m_n_samples;

    if (flag & db::BinFile::append) {
      error << "Cannot append data in read only mode." << std::endl;
      throw Exception();
    }
  }
  else
  {
    error << "Invalid combination of flags." << std::endl;
    throw Exception();
  }
}

db::BinFile::~BinFile() {
  close();
}

void db::BinFile::close() {
  // Rewrite the header and update the number of samples
  m_header.m_n_samples = m_n_arrays_written;
  if(m_openmode & db::BinFile::out) m_header.write(m_stream);

  m_stream.close();
}

void db::BinFile::initHeader(const Torch::core::array::ElementType type, 
    size_t ndim, const size_t* shape) {
  // Check that data have not already been written
  if (m_n_arrays_written > 0 ) {
    Torch::core::error << "Cannot init the header of an output stream in which data" <<
      " have already been written." << std::endl;
    throw Torch::core::Exception();
  }

  // Initialize header
  m_header.m_elem_type = type;
  m_header.m_elem_sizeof = Torch::core::array::getElementSize(type);
  m_header.setShape(ndim, shape);
  m_header.write(m_stream);
  m_header_init = true;
}

template <typename T>
void write_inlined(const db::detail::InlinedArrayImpl& data) {
  switch(data.getNDim()) {
    case 1:
      {
        const blitz::Array<T,1>& bz = data.get<T,1>();
        for (size_t i=0; i<data.getShape()[0]; ++i)
          m_stream.write(data(i));
      }
    case 2:
      {
        const blitz::Array<T,2>& bz = data.get<T,2>();
        for (size_t i=0; i<data.getShape()[0]; ++i)
          for (size_t j=0; j<data.getShape()[1]; ++j)
            m_stream.write(data(i,j));
      }
    case 3:
      {
        const blitz::Array<T,3>& bz = data.get<T,3>();
        for (size_t i=0; i<data.getShape()[0]; ++i)
          for (size_t j=0; j<data.getShape()[1]; ++j)
            for (size_t k=0; k<data.getShape()[2]; ++k)
              m_stream.write(data(i,j,k));
      }
    case 4:
      {
        const blitz::Array<T,4>& bz = data.get<T,4>();
        for (size_t i=0; i<data.getShape()[0]; ++i)
          for (size_t j=0; j<data.getShape()[1]; ++j)
            for (size_t k=0; k<data.getShape()[2]; ++k)
              for (size_t l=0; l<data.getShape()[3]; ++l)
              m_stream.write(data(i,j,k,l));
      }
    default:
      throw DimensionError();
  }
}

void db::BinFile::write(const db::detail::InlinedArrayImpl& data) {
  if(!m_header_init) {
    //initializes the header
    initHeader(data.getElementType(), data.getNDim(), data.getShape());
    else {
      //checks compatibility with previously written stuff
      if (data.getNDim() != m_header.getNDim()) throw DimensionError();
      const size_t* p_shape = data.getShape();
      const size_t* h_shape = m_header.getShape();
      for (size_t i=0; i<data.getNDim(); ++i)
        if(p_shape[i] != h_shape[i]) throw DimensionError();
    }
  }

  // copy the data into the output stream
  switch(data.getElementType()) {
    case Torch::core::array::t_bool: write_inlined<bool>(data); break;
    case Torch::core::array::t_int8: write_inlined<int8_t>(data); break;
    case Torch::core::array::t_int16: write_inlined<int16_t>(data); break;
    case Torch::core::array::t_int32: write_inlined<int32_t>(data); break;
    case Torch::core::array::t_int64: write_inlined<int64_t>(data); break;
    case Torch::core::array::t_uint8: write_inlined<uint8_t>(data); break;
    case Torch::core::array::t_uint16: write_inlined<uint16_t>(data); break;
    case Torch::core::array::t_uint32: write_inlined<uint32_t>(data); break;
    case Torch::core::array::t_uint64: write_inlined<uint64_t>(data); break;
    case Torch::core::array::t_float32: write_inlined<float>(data); break;
    case Torch::core::array::t_float64: write_inlined<double>(data); break;
    case Torch::core::array::t_float128: write_inlined<long double>(data); break;
    case Torch::core::array::t_complex64: write_inlined<std::complex<float> >(data); break;
    case Torch::core::array::t_complex128: write_inlined<std::complex<double> >(data); break;
    case Torch::core::array::t_complex256: write_inlined<std::complex<long double> >(data); break;
    default: throw TypeError();
  }

  // increment m_n_arrays_written and m_current_array
  ++m_current_array;
  if (m_current_array>m_n_arrays_written) ++m_n_arrays_written;
}

template <typename T>
db::detail::InlinedArrayImpl read_inlined(ndim, const size_t* shape) {
  switch(ndim) {
    case 1:
      {
        blitz::Array<T,1> bz(shape[0]);
        m_stream.read(reinterpret_cast<char*>(bz.data()), shape[0]*sizeof(T));
        return bz;
      }
    case 2:
      {
        //arrays are always stored in C-style ordering
        blitz::Array<T,1> bz(shape[0], shape[1]);
        m_stream.read(reinterpret_cast<char*>(bz.data()), shape[0]*shape[1]*sizeof(T));
        return bz;
      }
    case 3:
      {
        //arrays are always stored in C-style ordering
        blitz::Array<T,1> bz(shape[0], shape[1], shape[2]);
        m_stream.read(reinterpret_cast<char*>(bz.data()), shape[0]*shape[1]*shape[2]*sizeof(T));
        return bz;
      }
    case 4:
      {
        //arrays are always stored in C-style ordering
        blitz::Array<T,1> bz(shape[0], shape[1], shape[2], shape[3]);
        m_stream.read(reinterpret_cast<char*>(bz.data()), shape[0]*shape[1]*shape[2]*shape[3]*sizeof(T));
        return bz;
      }
    default:
      throw DimensionError();
  }
}

db::detail::InlinedArrayImpl db::BinFile::read() {
  if(!m_header_init) throw Uninitialized();
  switch(data.getElementType()) {
    case Torch::core::array::t_bool: return read_inlined<bool>();
    case Torch::core::array::t_int8: return read_inlined<int8_t>();
    case Torch::core::array::t_int16: return read_inlined<int16_t>();
    case Torch::core::array::t_int32: return read_inlined<int32_t>();
    case Torch::core::array::t_int64: return read_inlined<int64_t>();
    case Torch::core::array::t_uint8: return read_inlined<uint8_t>();
    case Torch::core::array::t_uint16: return read_inlined<uint16_t>();
    case Torch::core::array::t_uint32: return read_inlined<uint32_t>();
    case Torch::core::array::t_uint64: return read_inlined<uint64_t>();
    case Torch::core::array::t_float32: return read_inlined<float>();
    case Torch::core::array::t_float64: return read_inlined<double>();
    case Torch::core::array::t_float128: return read_inlined<long double>();
    case Torch::core::array::t_complex64: return read_inlined<std::complex<float> >();
    case Torch::core::array::t_complex128: return read_inlined<std::complex<double> >();
    case Torch::core::array::t_complex256: return read_inlined<std::complex<long double> >();
    default: throw TypeError();
  }
  ++m_current_array;
}

db::detail::InlinedArrayImpl db::BinFile::read (size_t index) {
  // Check that we are reaching an existing array
  if( index > m_header.m_n_samples ) {
    error << "Trying to reach a non-existing array." << std::endl;
    throw IndexError();
  }

  // Set the stream pointer at the correct position
  m_stream.seekg( m_header.getArrayIndex(index) );
  m_current_array = index;

  // Put the content of the stream in the blitz array.
  return read();
}
