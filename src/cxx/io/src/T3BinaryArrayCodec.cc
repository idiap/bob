/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements a Torch3vision bindata reader/writer
 *
 * The format, as described in the old source code goes like this.
 *
 * 1) data is always recorded in little endian format
 * 2) the first 4 bytes describe an integer that indicates the number of arrays
 * to follow
 * 3) the second 4 bytes describe an integer that specifies the frame width.
 * 4) all arrays inserted there are single dimensional arrays.
 * 5) all elements from all arrays are "normally" float (4-bytes), but could be
 * double if set in the header of T3 during compilation. The file size will
 * indicate the right type to use.
 *
 * Because of this restriction, this codec will only be able to work with
 * single-dimension input.
 */

#include "core/array_check.h"
#include "io/T3BinaryArrayCodec.h"
#include "io/ArrayCodecRegistry.h"
#include "io/Exception.h"
#include <fstream>

//some infrastructure to check the file size
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace io = Torch::io;

static inline size_t get_filesize(const std::string& filename) {
  struct stat filestatus;
  stat(filename.c_str(), &filestatus);
  return filestatus.st_size;
}

//Takes care of the codec registration.
static bool register_codec() {
  io::ArrayCodecRegistry::addCodec(boost::shared_ptr<io::ArrayCodec>(new io::T3BinaryArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

io::T3BinaryArrayCodec::T3BinaryArrayCodec()
  : m_name("torch3.array.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".bindata");
}

io::T3BinaryArrayCodec::~T3BinaryArrayCodec() { }

void io::T3BinaryArrayCodec::peek(const std::string& filename, 
    typeinfo& info) const {

  size_t fsize = get_filesize(filename);
  fsize -= 8; // remove the first two entries
  // read the first two 4-byte integers in the file, convert to unsigned
  std::ifstream ifile(filename.c_str(), std::ios::binary|std::ios::in);
  if (!ifile) throw io::FileNotReadable(filename);
  uint32_t nsamples, framesize;
  nsamples = framesize = 0;
  ifile.read((char*)&nsamples, sizeof(uint32_t));
  ifile.read((char*)&framesize, sizeof(uint32_t));
  ifile.close();
  // are those floats or doubles?
  if (fsize == (nsamples*framesize*sizeof(float))) 
    info.dtype = Torch::core::array::t_float32;
  else if (fsize == (nsamples*framesize*sizeof(double))) 
    info.dtype = Torch::core::array::t_float64;
  else 
    throw io::TypeError(Torch::core::array::t_float32, Torch::core::array::t_unknown);
  size_t shape[2] = {nsamples, framesize};
  info.set_shape<size_t>(2, &shape[0]);

}

void io::T3BinaryArrayCodec::load(const std::string& filename, 
    buffer& array) const {

  io::typeinfo info;
  peek(filename, info);
  if(!array.type().is_compatible(info)) array.set(info);

  //open the file, now for reading the contents...
  std::ifstream ifile(filename.c_str(), std::ios::binary|std::ios::in);

  //skip the first 8 bytes, that contain the header that we already read
  ifile.seekg(8);
  ifile.read(static_cast<char*>(array.ptr()), info.buffer_size());

}

void io::T3BinaryArrayCodec::save (const std::string& filename, 
    const buffer& array) const {

  const io::typeinfo& info = array.type();

  //can only save uni-dimensional data, so throw if that is not the case
  if (info.nd != 2) throw io::DimensionError(info.nd, 2);

  //can only save float32 or float64, otherwise, throw.
  if ((info.dtype != Torch::core::array::t_float32) && 
      (info.dtype != Torch::core::array::t_float64)) {
    throw io::UnsupportedTypeError(info.dtype); 
  }

  std::ofstream ofile(filename.c_str(), std::ios::binary|std::ios::out);

  //header writing...
  const uint32_t nsamples = info.shape[0];
  const uint32_t framesize = info.shape[1];
  ofile.write((const char*)&nsamples, sizeof(uint32_t));
  ofile.write((const char*)&framesize, sizeof(uint32_t));
  ofile.write(static_cast<const char*>(array.ptr()), info.buffer_size());

}
