/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon  7 Feb 21:34:22 2011 
 *
 * @brief Implements the BinaryArrayCodec type 
 */

#include "io/BinaryArrayCodec.h"
#include "io/BinFile.h"
#include "io/ArrayCodecRegistry.h"
#include "io/Exception.h"

namespace io = Torch::io;

//Takes care of the codec registration.
static bool register_codec() {
  io::ArrayCodecRegistry::addCodec(boost::shared_ptr<io::ArrayCodec>(new io::BinaryArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec(); 

io::BinaryArrayCodec::BinaryArrayCodec()
  : m_name("torch.array.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".bin");
}

io::BinaryArrayCodec::~BinaryArrayCodec() { }

void io::BinaryArrayCodec::peek(const std::string& file, io::typeinfo& info) const {
  io::BinFile f(file, io::BinFile::in);
  if (!f) {
    info.reset();
    throw io::FileNotReadable(file);
  }
  info.set(f.getElementType(), f.getNDimensions(), f.getShape());
}

void io::BinaryArrayCodec::load(const std::string& file,
    io::buffer& array) const {
  io::BinFile f(file, io::BinFile::in);
  if (!f) throw io::FileNotReadable(file);
  return f.read(array);
}

void io::BinaryArrayCodec::save (const std::string& file, 
    const io::buffer& array) const {
  io::BinFile f(file, io::BinFile::out);
  f.write(array);
}
