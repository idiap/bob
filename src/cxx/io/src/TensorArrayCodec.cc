/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Mon  7 Feb 21:34:22 2011 
 *
 * @brief Implements the TensorArrayCodec type 
 */

#include "io/TensorArrayCodec.h"
#include "io/TensorFile.h"
#include "io/ArrayCodecRegistry.h"
#include "io/Exception.h"

namespace io = Torch::io;

//Takes care of the codec registration.
static bool register_codec() {
  io::ArrayCodecRegistry::addCodec(boost::shared_ptr<io::ArrayCodec>(new io::TensorArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec(); 

io::TensorArrayCodec::TensorArrayCodec()
  : m_name("torch.array.tensor"),
    m_extensions()
{ 
  m_extensions.push_back(".tensor");
}

io::TensorArrayCodec::~TensorArrayCodec() { }

void io::TensorArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const {
  io::TensorFile f(filename, io::TensorFile::in);
  if (!f) {
    eltype = Torch::core::array::t_unknown;
    ndim = 0;
    throw io::FileNotReadable(filename);
  }
  eltype = f.getElementType();
  ndim = f.getNDimensions();
  for (size_t i=0; i<ndim; ++i) shape[i] = f.getShape()[i]; 
}

io::detail::InlinedArrayImpl 
io::TensorArrayCodec::load(const std::string& filename) const {
  io::TensorFile f(filename, io::TensorFile::in);
  if (!f) throw io::FileNotReadable(filename);
  return f.read();
}

void io::TensorArrayCodec::save (const std::string& filename, 
    const io::detail::InlinedArrayImpl& data) const {
  io::TensorFile f(filename, io::TensorFile::out);
  f.write(data);
}
