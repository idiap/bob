/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Mon  7 Feb 21:34:22 2011 
 *
 * @brief Implements the TensorArrayCodec type 
 */

#include "database/TensorArrayCodec.h"
#include "database/TensorFile.h"
#include "database/ArrayCodecRegistry.h"
#include "database/Exception.h"

namespace db = Torch::database;

//Takes care of the codec registration.
static bool register_codec() {
  db::ArrayCodecRegistry::addCodec(boost::shared_ptr<db::ArrayCodec>(new db::TensorArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec(); 

db::TensorArrayCodec::TensorArrayCodec()
  : m_name("torch.array.tensor"),
    m_extensions()
{ 
  m_extensions.push_back(".tensor");
}

db::TensorArrayCodec::~TensorArrayCodec() { }

void db::TensorArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const {
  db::TensorFile f(filename, db::TensorFile::in);
  if (!f) {
    eltype = Torch::core::array::t_unknown;
    ndim = 0;
    throw db::FileNotReadable(filename);
  }
  eltype = f.getElementType();
  ndim = f.getNDimensions();
  for (size_t i=0; i<ndim; ++i) shape[i] = f.getShape()[i]; 
}

db::detail::InlinedArrayImpl 
db::TensorArrayCodec::load(const std::string& filename) const {
  db::TensorFile f(filename, db::TensorFile::in);
  if (!f) throw db::FileNotReadable(filename);
  return f.read();
}

void db::TensorArrayCodec::save (const std::string& filename, 
    const db::detail::InlinedArrayImpl& data) const {
  db::TensorFile f(filename, db::TensorFile::out);
  f.write(data);
}
