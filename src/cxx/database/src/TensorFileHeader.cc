/**
 * @file src/cxx/database/database/TensorFileHeader.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * This class defines an header for storing multiarrays into .tensor files.
 */

#include "core/logging.h"

#include "database/TensorFileHeader.h"
#include "database/Exception.h"

namespace db = Torch::database;
namespace dbd = db::detail;
namespace core = Torch::core;

dbd::TensorFileHeader::TensorFileHeader()
  : m_tensor_type(db::Char),
    m_elem_type(core::array::t_unknown), 
    m_n_samples(0),
    m_n_dimensions(0), 
    m_tensor_size(0)
{
  for (size_t i=0; i<core::array::N_MAX_DIMENSIONS_ARRAY; ++i) m_shape[i] = 0;
}

dbd::TensorFileHeader::~TensorFileHeader() { }

size_t dbd::TensorFileHeader::getArrayIndex (size_t index) const {
  size_t header_size = 7 * sizeof(int);
  return header_size + index * m_tensor_size;
}

size_t dbd::TensorFileHeader::getSize(size_t dim_index) const {
  if(dim_index >= m_n_dimensions) throw db::DimensionError(dim_index, m_n_dimensions);
  return m_shape[dim_index]; 
}

void dbd::TensorFileHeader::read(std::istream& str) {
  // Start reading at the beginning of the stream
  str.seekg(std::ios_base::beg);

  int val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_tensor_type = (db::TensorType)val;
  m_elem_type = db::tensorTypeToArrayType(m_tensor_type);
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_n_samples = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_n_dimensions = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_shape[0] = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_shape[1] = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_shape[2] = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_shape[3] = (size_t)val;

  header_ok();
}

void dbd::TensorFileHeader::write(std::ostream& str) const
{
  // Start writing at the beginning of the stream
  str.seekp(std::ios_base::beg);

  int val;
  val = (int)m_tensor_type;
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_n_samples;
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_n_dimensions;
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_shape[0];
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_shape[1];
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_shape[2];
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_shape[3];
  str.write( reinterpret_cast<char*>(&val), sizeof(int));
}

void dbd::TensorFileHeader::header_ok()
{
  // Check the type
  switch (m_tensor_type)
  {
    // supported tensor types
    case db::Char:
    case db::Short:
    case db::Int:
    case db::Long:
    case db::Float:
    case db::Double:
      break;
    // error
    default:
      throw Torch::database::UnsupportedTypeError(Torch::core::array::t_unknown);
  }

  // Check the number of samples and dimensions
  if( m_n_samples < 0 || m_n_dimensions < 1 ||
      m_n_dimensions > 4)
    throw Torch::database::DimensionError(m_n_dimensions,4);

  // OK
  update();
}

void dbd::TensorFileHeader::update()
{
  size_t base_size = 0;
  switch (m_tensor_type)
  {
    case db::Char:    base_size = sizeof(char); break;
    case db::Short:   base_size = sizeof(short); break;
    case db::Int:     base_size = sizeof(int); break;
    case db::Long:    base_size = sizeof(long); break;
    case db::Float:   base_size = sizeof(float); break;
    case db::Double:  base_size = sizeof(double); break;
    default:
      throw Torch::database::UnsupportedTypeError(Torch::core::array::t_unknown);
  }

  size_t tsize = 1;
  for(size_t i = 0; i < m_n_dimensions; ++i)
    tsize *= m_shape[i];

  m_tensor_size = tsize * base_size;
}


db::TensorType db::arrayTypeToTensorType(Torch::core::array::ElementType eltype)
{
  switch(eltype)
  {
    case Torch::core::array::t_int8:
      return db::Char;
    case Torch::core::array::t_int16:
      return db::Short;
    case Torch::core::array::t_int32:
      return db::Int;
    case Torch::core::array::t_int64:
      return db::Long;
    case Torch::core::array::t_float32:
      return db::Float;
    case Torch::core::array::t_float64:
      return db::Double;
    default:
      throw Torch::database::UnsupportedTypeError(Torch::core::array::t_unknown);
  }
}
  
Torch::core::array::ElementType db::tensorTypeToArrayType(db::TensorType tensortype)
{
  switch(tensortype)
  {
    case db::Char:
      return Torch::core::array::t_int8;
    case db::Short:
      return Torch::core::array::t_int16;
    case db::Int:
      return Torch::core::array::t_int32;
    case db::Long:
      return Torch::core::array::t_int64;
    case db::Float:
      return Torch::core::array::t_float32;
    case db::Double:
      return Torch::core::array::t_float64;
    default:
      throw Torch::database::UnsupportedTypeError(Torch::core::array::t_unknown);
  }
}
