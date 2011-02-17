/**
 * @file src/cxx/database/database/BinFileHeader.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * This class defines an header for storing multiarrays into binary files.
 */

#include "core/logging.h"

#include "database/BinFileHeader.h"
#include "database/Exception.h"

namespace db = Torch::database;
namespace dbd = db::detail;
namespace core = Torch::core;

const uint32_t dbd::MAGIC_ENDIAN_DW = 0x01020304;
const uint8_t dbd::FORMAT_VERSION = 0;

dbd::BinFileHeader::BinFileHeader()
  : m_version(dbd::FORMAT_VERSION),
    m_elem_type(core::array::t_unknown), 
    m_elem_sizeof(0),//core::array::getElementSize(m_elem_type)),
    m_endianness(dbd::MAGIC_ENDIAN_DW),
    m_n_dimensions(0), 
    m_n_samples(0)
{
  for (size_t i=0; i<core::array::N_MAX_DIMENSIONS_ARRAY; ++i) m_shape[i] = 0;
}

dbd::BinFileHeader::~BinFileHeader() { }

size_t dbd::BinFileHeader::getArrayIndex (size_t index) const {
  size_t header_size = 4*sizeof(uint8_t) + sizeof(uint32_t)
    + (1+m_n_dimensions)*sizeof(uint64_t);
  return header_size + index * getNElements() * m_elem_sizeof;
}

size_t dbd::BinFileHeader::getSize (size_t dim_index) const {
  if(dim_index >= m_n_dimensions) throw db::DimensionError(dim_index, m_n_dimensions);
  return m_shape[dim_index]; 
}

void dbd::BinFileHeader::read (std::istream& str) {
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
  m_elem_type = static_cast<core::array::ElementType>(val8);
  TDEBUG3("Array-type: " << m_elem_type);
  // call function to update other type-related member (m_data_size_of)

  // Element sizeof
  // Check that the value stored in the header matches the run-time value
  str.read( reinterpret_cast<char*>(&val8), sizeof(uint8_t));
  m_elem_sizeof = static_cast<uint8_t>(val8);
  size_t runtime_sizeof = core::array::getElementSize(m_elem_type);
  if( runtime_sizeof != m_elem_sizeof )
    core::warn << "The size of the element type stored in the header does" <<
      " not match the runtime size." << std::endl;
  TDEBUG3("Sizeof: " << m_elem_sizeof);

  // Number of dimensions
  str.read (reinterpret_cast<char*>(&val8), sizeof(uint8_t));
  m_n_dimensions = static_cast<uint8_t>(val8);
  if( m_n_dimensions > core::array::N_MAX_DIMENSIONS_ARRAY) {
    throw db::DimensionError(m_n_dimensions, core::array::N_MAX_DIMENSIONS_ARRAY);
  }
  TDEBUG3("Number of dimensions: " << m_n_dimensions);

  // Endianness
  str.read (reinterpret_cast<char*>(&val32), sizeof(uint32_t));
  if(val32 != dbd::MAGIC_ENDIAN_DW) {
    core::error << "The data has been saved on a machine with a different " <<
      " endianness." << std::endl;
    throw core::Exception();
  }
  m_endianness = static_cast<uint32_t>(val32);
  TDEBUG3("Endianness: " << m_endianness);

  // Size of each dimension
  for( size_t i=0; i<core::array::N_MAX_DIMENSIONS_ARRAY; ++i) {
    if( i<m_n_dimensions) {
      str.read( reinterpret_cast<char*>(&val64), sizeof(uint64_t));
      m_shape[i] = static_cast<uint64_t>(val64);
    }
    else
      m_shape[i] = 0;
    TDEBUG3("  Dimension " << i << ": " << m_shape[i]);
  }
  
  // Number of samples
  str.read( reinterpret_cast<char*>(&val64), sizeof(uint64_t));
  m_n_samples = static_cast<uint64_t>(val64);
  TDEBUG3("Number of samples: " << m_n_samples);
}

void dbd::BinFileHeader::write(std::ostream& str) const
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
