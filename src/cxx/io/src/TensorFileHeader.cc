/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * This class defines an header for storing multiarrays into .tensor files.
 */

#include <boost/format.hpp>
#include "core/logging.h"

#include "io/TensorFileHeader.h"
#include "io/Exception.h"

namespace io = Torch::io;
namespace iod = io::detail;
namespace core = Torch::core;

iod::TensorFileHeader::TensorFileHeader()
  : m_tensor_type(io::Char),
    m_type(),
    m_n_samples(0),
    m_tensor_size(0)
{
}

iod::TensorFileHeader::~TensorFileHeader() { }

size_t iod::TensorFileHeader::getArrayIndex (size_t index) const {
  size_t header_size = 7 * sizeof(int);
  return header_size + index * m_tensor_size;
}

void iod::TensorFileHeader::read(std::istream& str) {
  // Start reading at the beginning of the stream
  str.seekg(std::ios_base::beg);

  int val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_tensor_type = (io::TensorType)val;
  m_type.dtype = io::tensorTypeToArrayType(m_tensor_type);

  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  m_n_samples = (size_t)val;
  
  int nd;
  str.read(reinterpret_cast<char*>(&nd), sizeof(int));

  int shape[TORCH_MAX_DIM];
  
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

void iod::TensorFileHeader::write(std::ostream& str) const
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

void iod::TensorFileHeader::header_ok()
{
  // Check the type
  switch (m_tensor_type)
  {
    // supported tensor types
    case io::Char:
    case io::Short:
    case io::Int:
    case io::Long:
    case io::Float:
    case io::Double:
      break;
    // error
    default:
      throw io::UnsupportedTypeError(core::array::t_unknown);
  }

  // Check the number of samples and dimensions
  if( m_n_samples < 0 || m_type.nd < 1 || m_type.nd > 4)
    throw io::DimensionError(m_type.nd,4);

  // OK
  update();
}

void iod::TensorFileHeader::update()
{
  size_t base_size = 0;
  switch (m_tensor_type)
  {
    case io::Char:    base_size = sizeof(char); break;
    case io::Short:   base_size = sizeof(short); break;
    case io::Int:     base_size = sizeof(int); break;
    case io::Long:    base_size = sizeof(long); break;
    case io::Float:   base_size = sizeof(float); break;
    case io::Double:  base_size = sizeof(double); break;
    default:
      throw io::UnsupportedTypeError(core::array::t_unknown);
  }

  size_t tsize = 1;
  for(size_t i = 0; i < m_type.nd; ++i) tsize *= m_type.shape[i];

  m_tensor_size = tsize * base_size;
}


io::TensorType io::arrayTypeToTensorType(core::array::ElementType eltype)
{
  switch(eltype)
  {
    case core::array::t_int8:
      return io::Char;
    case core::array::t_int16:
      return io::Short;
    case core::array::t_int32:
      return io::Int;
    case core::array::t_int64:
      return io::Long;
    case core::array::t_float32:
      return io::Float;
    case core::array::t_float64:
      return io::Double;
    default:
      throw io::UnsupportedTypeError(core::array::t_unknown);
  }
}
  
core::array::ElementType io::tensorTypeToArrayType(io::TensorType tensortype)
{
  switch(tensortype)
  {
    case io::Char:
      return core::array::t_int8;
    case io::Short:
      return core::array::t_int16;
    case io::Int:
      return core::array::t_int32;
    case io::Long:
      return core::array::t_int64;
    case io::Float:
      return core::array::t_float32;
    case io::Double:
      return core::array::t_float64;
    default:
      throw io::UnsupportedTypeError(core::array::t_unknown);
  }
}
