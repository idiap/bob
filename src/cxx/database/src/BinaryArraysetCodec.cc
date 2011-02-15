/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon  7 Feb 21:34:22 2011 
 *
 * @brief Implements the BinaryArraysetCodec type 
 */

#include "database/BinaryArraysetCodec.h"
#include "database/ArraysetCodecRegistry.h"
#include "database/BinFile.h"

namespace db = Torch::database;

//Takes care of the codec registration.
static bool register_codec() {
  db::ArraysetCodecRegistry::addCodec(boost::shared_ptr<db::ArraysetCodec>(new db::BinaryArraysetCodec())); 
  return true;
}

static bool codec_registered = register_codec(); 

db::BinaryArraysetCodec::BinaryArraysetCodec()
  : m_name("torch.arrayset.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".bin");
}

db::BinaryArraysetCodec::~BinaryArraysetCodec() { }

void db::BinaryArraysetCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape, size_t& samples) const {
  db::BinFile f(filename, db::BinFile::in);
  if (!f) {
    eltype = Torch::core::array::t_unknown;
    ndim = 0;
    samples = 0;
    return;
  }
  eltype = f.getElementType();
  ndim = f.getNDimensions();
  samples = f.getNSamples();
  for (size_t i=0; i<ndim; ++i) shape[i] = f.getShape()[i]; 
}

db::detail::InlinedArraysetImpl db::BinaryArraysetCodec::load
(const std::string& filename) const {
  db::BinFile f(filename, db::BinFile::in);
  db::detail::InlinedArraysetImpl retval;
  for (size_t i=0; i<f.getNSamples(); ++i) retval.add(f.read());
  return retval;
}

db::Array db::BinaryArraysetCodec::load
(const std::string& filename, size_t id) const {
  db::BinFile f(filename, db::BinFile::in);
  return f.read(id-1);
}

void db::BinaryArraysetCodec::append
(const std::string& filename, const Array& array) const {
  db::BinFile f(filename, db::BinFile::out | db::BinFile::append);
  f.write(array.get());
}

void db::BinaryArraysetCodec::save (const std::string& filename, 
    const detail::InlinedArraysetImpl& data) const {
  db::BinFile f(filename, db::BinFile::out);
  for (std::map<size_t, boost::shared_ptr<db::Array> >::const_iterator it = data.index().begin(); it != data.index().end(); ++it) f.write(it->second.get()->get());
}
