/**
 * @file src/cxx/core/core/BinFileHeader.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class defines an header for storing multiarrays into
 * binary files.
 */

#include "core/BinFileHeader.h"

namespace Torch {
  namespace core {

    BinFileHeader::BinFileHeader():
      m_version(0), m_type(array::t_unknown), m_n_dimensions(0), 
      m_n_samples(0), m_endianness(0), m_n_elements(0), m_data_sizeof(0)
    {
      for( size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i)
        m_shape[i] = 0;
    }

    size_t BinFileHeader::getArrayIndex(size_t index) const {
      size_t header_size = 5*sizeof(uint32_t) 
        + (1+m_n_dimensions)*sizeof(uint64_t);
      return header_size + index * m_n_elements * m_data_sizeof;
    }

    void BinFileHeader::sizeUpdated() {
      m_n_elements = m_shape[0];
      size_t i;
      for(i=1; i<array::N_MAX_DIMENSIONS_ARRAY && m_shape[i]!=0; ++i) {
        m_n_elements *= m_shape[i];
      }
      m_n_dimensions = i;
    }

    void BinFileHeader::typeUpdated() {
      size_t data_size;
      switch(m_type)
      {
        case array::t_bool:
          data_size = sizeof(bool); break;
        case array::t_int8:
          data_size = sizeof(int8_t); break;
        case array::t_int16:
          data_size = sizeof(int16_t); break;
        case array::t_int32:
          data_size = sizeof(int32_t); break;
        case array::t_int64:
          data_size = sizeof(int64_t); break;
        case array::t_uint8:
          data_size = sizeof(uint8_t); break;
        case array::t_uint16:
          data_size = sizeof(uint16_t); break;
        case array::t_uint32:
          data_size = sizeof(uint32_t); break;
        case array::t_uint64:
          data_size = sizeof(uint64_t); break;
        case array::t_float32:
          data_size = sizeof(float); break;
        case array::t_float64:
          data_size = sizeof(double); break;
        case array::t_float128:
          data_size = sizeof(long double); break;
        case array::t_complex64:
          data_size = sizeof(std::complex<float>); break;
        case array::t_complex128:
          data_size = sizeof(std::complex<double>); break;
        case array::t_complex256:
          data_size = sizeof(std::complex<long double>); break;
        default:
          error << "Unknown type." << std::endl;
          throw Exception();
          break;
      }
      m_data_sizeof = data_size;
    }


    size_t BinFileHeader::getSize(size_t dim_index) const { 
      if( dim_index>=array::N_MAX_DIMENSIONS_ARRAY)
        throw Exception();
      return m_shape[dim_index]; 
    }

    void BinFileHeader::setSize(size_t dim_index, size_t val) {
      if( dim_index>=array::N_MAX_DIMENSIONS_ARRAY)
        throw Exception();
      m_shape[dim_index] = val;
      sizeUpdated();
    }


    void BinFileHeader::read( std::istream& str)
    {
      // Start reading at the beginning of the stream
      str.seekg(std::ios_base::beg);

      // data is read from explicit types and converted back
      uint32_t val32;
      uint64_t val64;
      str.read( reinterpret_cast<char*>(&val32), sizeof(uint32_t));
      m_version = static_cast<size_t>(val32);
      TDEBUG3("Version: " << m_version);
      str.read( reinterpret_cast<char*>(&val32), sizeof(uint32_t));
      m_type = static_cast<array::ArrayType>(val32);
      TDEBUG3("Array-type: " << m_type);
      str.read( reinterpret_cast<char*>(&val32), sizeof(uint32_t));
      // call function to update other type-related member (m_data_size_of)
      typeUpdated();
      m_n_dimensions = static_cast<size_t>(val32);
      if( m_n_dimensions > array::N_MAX_DIMENSIONS_ARRAY) {
        error << "The number of dimensions is larger the maximal number " <<
          "of dimensions supported by this version of Torch5spro." << 
          std::endl;
        throw Exception();
      }
      TDEBUG3("Number of dimensions: " << m_n_dimensions);
      for( size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i) {
        if( i<m_n_dimensions) {
          str.read( reinterpret_cast<char*>(&val64), sizeof(uint64_t));
          m_shape[i] = static_cast<size_t>(val64);
        }
        else
          m_shape[i] = 0;
        TDEBUG3("  Dimension " << i << ": " << m_shape[i]);
      }
      // call function to update other size-related members
      sizeUpdated();

      str.read( reinterpret_cast<char*>(&val64), sizeof(uint64_t));
      m_n_samples = static_cast<size_t>(val64);
      TDEBUG3("Number of samples: " << m_n_samples);
      str.read( reinterpret_cast<char*>(&val32), sizeof(uint32_t));
      m_endianness = static_cast<size_t>(val32);
      TDEBUG3("Endianness: " << m_endianness);

      // Read the sizeof value stored in the header and check that it matches
      // the run-time value
      str.read( reinterpret_cast<char*>(&val32), sizeof(uint32_t));
      size_t runtime_sizeof;
      switch(m_type)
      {
        case array::t_bool:
          runtime_sizeof = sizeof(bool); break;
        case array::t_int8:
          runtime_sizeof = sizeof(int8_t); break;
        case array::t_int16:
          runtime_sizeof = sizeof(int16_t); break;
        case array::t_int32:
          runtime_sizeof = sizeof(int32_t); break;
        case array::t_int64:
          runtime_sizeof = sizeof(int64_t); break;
        case array::t_uint8:
          runtime_sizeof = sizeof(uint8_t); break;
        case array::t_uint16:
          runtime_sizeof = sizeof(uint16_t); break;
        case array::t_uint32:
          runtime_sizeof = sizeof(uint32_t); break;
        case array::t_uint64:
          runtime_sizeof = sizeof(uint64_t); break;
        case array::t_float32:
          runtime_sizeof = sizeof(float); break;
        case array::t_float64:
          runtime_sizeof = sizeof(double); break;
        case array::t_float128:
          runtime_sizeof = sizeof(long double); break;
        case array::t_complex64:
          runtime_sizeof = sizeof(std::complex<float>); break;
        case array::t_complex128:
          runtime_sizeof = sizeof(std::complex<double>); break;
        case array::t_complex256:
          runtime_sizeof = sizeof(std::complex<long double>); break;
        default:
          error << "Unknown type." << std::endl;
          throw Exception();
          break;
      }
      if( runtime_sizeof != m_data_sizeof )
        warn << "The size of the element type stored in the header does" <<
          " not match the runtime size. This might be the case with long " <<
          " double and std::complex<long double> when transmitting data " <<
          " between 32 bits and 64 bits Linux machines!" << std::endl;
      TDEBUG3("Sizeof: " << m_data_sizeof);
    }


    void BinFileHeader::write(std::ostream& str) const
    {
      // Start writing at the beginning of the stream
      str.seekp(std::ios_base::beg);

      // data is converted to more explicit types before being written 
      // in order to improve portability
      uint32_t uint32;
      uint64_t uint64;
      uint32 = static_cast<uint32_t>(m_version);
      str.write( reinterpret_cast<char*>(&uint32), sizeof(uint32_t) );
      TDEBUG3("Version: " << m_version);
      uint32 = static_cast<uint32_t>(m_type);
      str.write( reinterpret_cast<char*>(&uint32), sizeof(uint32_t) );
      TDEBUG3("Array-type: " << m_type);
      uint32 = static_cast<uint32_t>(m_n_dimensions);
      str.write( reinterpret_cast<char*>(&uint32), sizeof(uint32_t) );
      TDEBUG3("Number of dimensions: " << m_n_dimensions);
      for( size_t i=0; i<m_n_dimensions; ++i) {
        uint64 = static_cast<uint64_t>(m_shape[i]);
        str.write( reinterpret_cast<char*>(&uint64), sizeof(uint64_t) );
        TDEBUG3("  Dimension " << i << ": " << m_shape[i]);
      }
      uint64 = static_cast<uint64_t>(m_n_samples);
      str.write( reinterpret_cast<char*>(&uint64), sizeof(uint64_t) );
      TDEBUG3("Number of samples: " << m_n_samples);
      uint32 = static_cast<uint32_t>(m_endianness);
      str.write( reinterpret_cast<char*>(&uint32), sizeof(uint32_t) );
      TDEBUG3("Endianness: " << m_endianness);
      uint32 = static_cast<uint32_t>(m_data_sizeof);
      str.write( reinterpret_cast<char*>(&uint32), sizeof(uint32_t) );
      TDEBUG3("Sizeof: " << m_data_sizeof);
    }


  }
}

