/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon  7 Feb 21:34:22 2011 
 *
 * @brief Implements the BinaryArraysetCodec type 
 */

#include <boost/filesystem.hpp>

#include "io/BinaryArraysetCodec.h"
#include "io/ArraysetCodecRegistry.h"
#include "io/BinFile.h"

namespace io = Torch::io;

//Takes care of the codec registration.
static bool register_codec() {
  io::ArraysetCodecRegistry::addCodec(boost::shared_ptr<io::ArraysetCodec>(new io::BinaryArraysetCodec())); 
  return true;
}

static bool codec_registered = register_codec(); 

io::BinaryArraysetCodec::BinaryArraysetCodec()
  : m_name("torch.arrayset.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".bin");
}

io::BinaryArraysetCodec::~BinaryArraysetCodec() { }

void io::BinaryArraysetCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape, size_t& samples) const {
  io::BinFile f(filename, io::BinFile::in);
  if (!f) {
    eltype = Torch::core::array::t_unknown;
    ndim = 0;
    samples = 0;
    throw io::FileNotReadable(filename);
  }
  eltype = f.getElementType();
  ndim = f.getNDimensions();
  samples = f.getNSamples();
  for (size_t i=0; i<ndim; ++i) shape[i] = f.getShape()[i]; 
}

io::detail::InlinedArraysetImpl io::BinaryArraysetCodec::load
(const std::string& filename) const {
  if (! boost::filesystem::exists(filename)) throw io::FileNotReadable(filename); 
  io::BinFile f(filename, io::BinFile::in);
  io::detail::InlinedArraysetImpl retval;
  for (size_t i=0; i<f.getNSamples(); ++i) retval.add(f.read());
  return retval;
}

io::Array io::BinaryArraysetCodec::load
(const std::string& filename, size_t id) const {
  if (! boost::filesystem::exists(filename)) throw io::FileNotReadable(filename); 
  io::BinFile f(filename, io::BinFile::in);
  if (!f) throw io::FileNotReadable(filename);
  return f.read(id-1);
}

void io::BinaryArraysetCodec::append
(const std::string& filename, const Array& array) const {
  if (boost::filesystem::exists(filename)) { //real append
    io::BinFile f(filename, io::BinFile::out | io::BinFile::append);
    f.write(array.get());
  } 
  else { //new file
    io::BinFile f(filename, io::BinFile::out);
    f.write(array.get());
  }
}

void io::BinaryArraysetCodec::save (const std::string& filename, 
    const detail::InlinedArraysetImpl& data) const {
  io::BinFile f(filename, io::BinFile::out);
  for (std::map<size_t, boost::shared_ptr<io::Array> >::const_iterator it = data.index().begin(); it != data.index().end(); ++it) f.write(it->second.get()->get());
}
