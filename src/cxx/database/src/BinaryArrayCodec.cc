/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon  7 Feb 21:34:22 2011 
 *
 * @brief Implements the BinaryArrayCodec type 
 */

#include "database/BinaryArrayCodec.h"
#include "database/BinFile.h"
#include "database/ArrayCodecRegistry.h"

namespace db = Torch::database;

//Takes care of the codec registration.
static bool register_codec() {
  db::ArrayCodecRegistry::addCodec(new db::BinaryArrayCodec()); 
}

static bool codec_registered = register_codec(); 

db::BinaryArrayCodec::BinaryArrayCodec()
  : m_name("torch.array.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".bin");
}

db::BinaryArrayCodec::~BinaryArrayCodec() { }

void db::BinaryArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim
    size_t& shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY]) const {
  db::BinFile f(filename, db::BinFile::in);
  eltype = f.getElementType();
  ndim = f.getNDimensions();
  for (size_t i=0; i<ndim; ++i) shape[i] = f.getShape()[i]; 
}

db::detail::InlinedArrayImpl 
db::BinaryArrayCodec::load(const std::string& filename) const {
  db::BinFile f(filename, db::BinFile::in);
  eltype = f.getElementType();
  ndim = f.getNDimensions();
}

void db::BinaryArrayCodec::save (const std::string& filename, 
    const db::detail::InlinedArrayImpl& data) const {
  db::BinFile f(filename, db::BinFile::out);
  f.write(data);
}
