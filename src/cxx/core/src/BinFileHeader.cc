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

    const uint32_t BinaryFile::MAGIC_ENDIAN_DW = 0x01020304;
    const uint8_t BinaryFile::FORMAT_VERSION = 0;

    BinFileHeader::BinFileHeader():
      m_version(BinaryFile::FORMAT_VERSION), m_elem_type(array::t_unknown), 
      m_elem_sizeof(0), m_n_dimensions(0), 
      m_endianness(BinaryFile::MAGIC_ENDIAN_DW), m_n_samples(0), 
      m_n_elements(0)
    {
      for( size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i)
        m_shape[i] = 0;
    }

    size_t BinFileHeader::getArrayIndex(size_t index) const {
      size_t header_size = 4*sizeof(uint8_t) + sizeof(uint32_t)
        + (1+m_n_dimensions)*sizeof(uint64_t);
      return header_size + index * m_n_elements * m_elem_sizeof;
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
      switch(m_elem_type)
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
        case array::t_complex64:
          data_size = sizeof(std::complex<float>); break;
        case array::t_complex128:
          data_size = sizeof(std::complex<double>); break;
        default:
          error << "Unknown type." << std::endl;
          throw Exception();
          break;
      }
      m_elem_sizeof = data_size;
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
      uint8_t val8;
      uint32_t val32;
      uint64_t val64;

      // Version
      str.read( reinterpret_cast<char*>(&val8), sizeof(uint8_t));
      m_version = static_cast<uint8_t>(val8);
      TDEBUG3("Version: " << m_version);

      // Element type
      str.read( reinterpret_cast<char*>(&val8), sizeof(uint8_t));
      m_elem_type = static_cast<array::ElementType>(val8);
      TDEBUG3("Array-type: " << m_elem_type);
      // call function to update other type-related member (m_data_size_of)

      // Element sizeof
      // Check that the value stored in the header matches the run-time value
      str.read( reinterpret_cast<char*>(&val8), sizeof(uint8_t));
      m_elem_sizeof = static_cast<uint8_t>(val8);
      size_t runtime_sizeof;
      switch(m_elem_type)
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
        case array::t_complex64:
          runtime_sizeof = sizeof(std::complex<float>); break;
        case array::t_complex128:
          runtime_sizeof = sizeof(std::complex<double>); break;
        default:
          error << "Unknown type." << std::endl;
          throw Exception();
          break;
      }
      if( runtime_sizeof != m_elem_sizeof )
        warn << "The size of the element type stored in the header does" <<
          " not match the runtime size." << std::endl;
      TDEBUG3("Sizeof: " << m_elem_sizeof);

      // Number of dimensions
      str.read( reinterpret_cast<char*>(&val8), sizeof(uint8_t));
      m_n_dimensions = static_cast<uint8_t>(val8);
      if( m_n_dimensions > array::N_MAX_DIMENSIONS_ARRAY) {
        error << "The number of dimensions is larger the maximal number " <<
          "of dimensions supported by this version of Torch5spro." << 
          std::endl;
        throw Exception();
      }
      TDEBUG3("Number of dimensions: " << m_n_dimensions);

      // Endianness
      str.read( reinterpret_cast<char*>(&val32), sizeof(uint32_t));
      if(val32 != BinaryFile::MAGIC_ENDIAN_DW)
      {
        error << "The data has been saved on a machine with a different " <<
          " endianness." << std::endl;
        throw Exception();
      }
      m_endianness = static_cast<uint32_t>(val32);
      TDEBUG3("Endianness: " << m_endianness);

      // Size of each dimension
      for( size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i) {
        if( i<m_n_dimensions) {
          str.read( reinterpret_cast<char*>(&val64), sizeof(uint64_t));
          m_shape[i] = static_cast<uint64_t>(val64);
        }
        else
          m_shape[i] = 0;
        TDEBUG3("  Dimension " << i << ": " << m_shape[i]);
      }
      // call function to update other size-related members
      sizeUpdated();

      // Number of samples
      str.read( reinterpret_cast<char*>(&val64), sizeof(uint64_t));
      m_n_samples = static_cast<uint64_t>(val64);
      TDEBUG3("Number of samples: " << m_n_samples);
    }


    void BinFileHeader::write(std::ostream& str) const
    {
      // Start writing at the beginning of the stream
      str.seekp(std::ios_base::beg);

      // data is converted to more explicit types before being written 
      // in order to improve portability
      uint8_t uint8;
      uint32_t uint32;
      uint64_t uint64;

      // Version
      uint8 = static_cast<uint8_t>(m_version);
      str.write( reinterpret_cast<char*>(&uint8), sizeof(uint8_t) );
      TDEBUG3("Version: " << m_version);
      
      // Element type
      uint8 = static_cast<uint8_t>(m_elem_type);
      str.write( reinterpret_cast<char*>(&uint8), sizeof(uint8_t) );
      TDEBUG3("Array-type: " << m_elem_type);

      // Element sizeof
      uint8 = static_cast<uint8_t>(m_elem_sizeof);
      str.write( reinterpret_cast<char*>(&uint8), sizeof(uint8_t) );
      TDEBUG3("Sizeof: " << m_elem_sizeof);

      // Number of dimensions
      uint8 = static_cast<uint8_t>(m_n_dimensions);
      str.write( reinterpret_cast<char*>(&uint8), sizeof(uint8_t) );
      TDEBUG3("Number of dimensions: " << m_n_dimensions);

      // Endianness
      uint32 = static_cast<uint32_t>(m_endianness);
      str.write( reinterpret_cast<char*>(&uint32), sizeof(uint32_t) );
      TDEBUG3("Endianness: " << m_endianness);

      // Size of each dimension
      for( size_t i=0; i<m_n_dimensions; ++i) {
        uint64 = static_cast<uint64_t>(m_shape[i]);
        str.write( reinterpret_cast<char*>(&uint64), sizeof(uint64_t) );
        TDEBUG3("  Dimension " << i << ": " << m_shape[i]);
      }
      
      // Number of samples
      uint64 = static_cast<uint64_t>(m_n_samples);
      str.write( reinterpret_cast<char*>(&uint64), sizeof(uint64_t) );
      TDEBUG3("Number of samples: " << m_n_samples);
    }


/************** Full specialization definitions *************/
#define NEED_CAST_DEF(T,name,D) template<> \
    bool BinFileHeader::needCast(const blitz::Array<T,D>& bl) const \
    {\
      std::cout << "Not Generic " << name << std::endl;\
      if(m_elem_type == name )\
        return false;\
      return true;\
    }\

    NEED_CAST_DEF(bool,array::t_bool,1)
    NEED_CAST_DEF(bool,array::t_bool,2)
    NEED_CAST_DEF(bool,array::t_bool,3)
    NEED_CAST_DEF(bool,array::t_bool,4)
    NEED_CAST_DEF(int8_t,array::t_int8,1)
    NEED_CAST_DEF(int8_t,array::t_int8,2)
    NEED_CAST_DEF(int8_t,array::t_int8,3)
    NEED_CAST_DEF(int8_t,array::t_int8,4)
    NEED_CAST_DEF(int16_t,array::t_int16,1)
    NEED_CAST_DEF(int16_t,array::t_int16,2)
    NEED_CAST_DEF(int16_t,array::t_int16,3)
    NEED_CAST_DEF(int16_t,array::t_int16,4)
    NEED_CAST_DEF(int32_t,array::t_int32,1)
    NEED_CAST_DEF(int32_t,array::t_int32,2)
    NEED_CAST_DEF(int32_t,array::t_int32,3)
    NEED_CAST_DEF(int32_t,array::t_int32,4)
    NEED_CAST_DEF(int64_t,array::t_int64,1)
    NEED_CAST_DEF(int64_t,array::t_int64,2)
    NEED_CAST_DEF(int64_t,array::t_int64,3)
    NEED_CAST_DEF(int64_t,array::t_int64,4)
    NEED_CAST_DEF(uint8_t,array::t_uint8,1)
    NEED_CAST_DEF(uint8_t,array::t_uint8,2)
    NEED_CAST_DEF(uint8_t,array::t_uint8,3)
    NEED_CAST_DEF(uint8_t,array::t_uint8,4)
    NEED_CAST_DEF(uint16_t,array::t_uint16,1)
    NEED_CAST_DEF(uint16_t,array::t_uint16,2)
    NEED_CAST_DEF(uint16_t,array::t_uint16,3)
    NEED_CAST_DEF(uint16_t,array::t_uint16,4)
    NEED_CAST_DEF(uint32_t,array::t_uint32,1)
    NEED_CAST_DEF(uint32_t,array::t_uint32,2)
    NEED_CAST_DEF(uint32_t,array::t_uint32,3)
    NEED_CAST_DEF(uint32_t,array::t_uint32,4)
    NEED_CAST_DEF(uint64_t,array::t_uint64,1)
    NEED_CAST_DEF(uint64_t,array::t_uint64,2)
    NEED_CAST_DEF(uint64_t,array::t_uint64,3)
    NEED_CAST_DEF(uint64_t,array::t_uint64,4)
    NEED_CAST_DEF(float,array::t_float32,1)
    NEED_CAST_DEF(float,array::t_float32,2)
    NEED_CAST_DEF(float,array::t_float32,3)
    NEED_CAST_DEF(float,array::t_float32,4)
    NEED_CAST_DEF(double,array::t_float64,1)
    NEED_CAST_DEF(double,array::t_float64,2)
    NEED_CAST_DEF(double,array::t_float64,3)
    NEED_CAST_DEF(double,array::t_float64,4)
    NEED_CAST_DEF(std::complex<float>,array::t_complex64,1)
    NEED_CAST_DEF(std::complex<float>,array::t_complex64,2)
    NEED_CAST_DEF(std::complex<float>,array::t_complex64,3)
    NEED_CAST_DEF(std::complex<float>,array::t_complex64,4)
    NEED_CAST_DEF(std::complex<double>,array::t_complex128,1)
    NEED_CAST_DEF(std::complex<double>,array::t_complex128,2)
    NEED_CAST_DEF(std::complex<double>,array::t_complex128,3)
    NEED_CAST_DEF(std::complex<double>,array::t_complex128,4)

  }
}

