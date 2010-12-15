/**
 * @file src/core/core/BinFileHeader.cc
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
      m_n_samples(0), m_endianness(0), m_data_size(0)
    {
      for( size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i)
        m_shape[i] = 0;
    }

    size_t BinFileHeader::getArrayIndex(size_t index) const {
      size_t header_size = 4*sizeof(uint32_t) 
        + (1+m_n_dimensions)*sizeof(uint64_t);
      // TODO: sizeof( type used)
      return header_size + index * m_data_size;
    }

    void BinFileHeader::updateSize() {
      m_data_size = m_shape[0];
      size_t i = 1;
      while( i < array::N_MAX_DIMENSIONS_ARRAY && m_shape[i] != 0) {
        m_data_size *= m_shape[i];
        ++i;
      }
      m_n_dimensions = (m_shape[0]!=0 ? i : 0);
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
      updateSize();
    }


    void BinFileHeader::read( std::istream& str)
    {
      // Start reading at the beginning of the stream
      str.seekg(std::ios_base::beg);

      // data is read from explicit types and converted back
      uint32_t val32;
      uint64_t val64;
      str >> val32;
      m_version = static_cast<size_t>(val32);
      str >> val32;
      m_type = static_cast<array::ArrayType>(val32);
      str >> val32;
      m_n_dimensions = static_cast<size_t>(val32);
      if( m_n_dimensions > array::N_MAX_DIMENSIONS_ARRAY) {
        error << "The number of dimensions is larger the maximal number " <<
          "of dimensions supported by this version of Torch5spro" << 
          std::endl;
        throw Exception();
      }
      for( size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i) {
        if( i<m_n_dimensions) {
          str >> val64;
          m_shape[i] = static_cast<size_t>(val64);
        }
        else
          m_shape[i] = 0;
      }
      str >> val64;
      m_n_samples = static_cast<size_t>(val64);
      str >> val32;
      m_endianness = static_cast<size_t>(val32);
    }


    void BinFileHeader::write(std::ostream& str) const
    {
      // Start writing at the beginning of the stream
      str.seekp(std::ios_base::beg);

      // data is converted to more explicit types before being written 
      // in order to improve portability
      str << static_cast<uint32_t>(m_version);
      str << static_cast<uint32_t>(m_type);
      str << static_cast<uint32_t>(m_n_dimensions);
      for( size_t i=0; i<m_n_dimensions; ++i)
        str << static_cast<uint64_t>(m_shape[i]);
      str << static_cast<uint64_t>(m_n_samples);
      str << static_cast<uint32_t>(m_endianness);
    }


  }
}

