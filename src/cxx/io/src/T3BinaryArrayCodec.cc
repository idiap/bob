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
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const {
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
  if (fsize == (nsamples*framesize*sizeof(float))) eltype = Torch::core::array::t_float32;
  else if (fsize == (nsamples*framesize*sizeof(double))) eltype = Torch::core::array::t_float64;
  else throw io::TypeError(Torch::core::array::t_float32, Torch::core::array::t_unknown);
  ndim = 2;
  shape[0] = nsamples;
  shape[1] = framesize;
}

io::detail::InlinedArrayImpl 
io::T3BinaryArrayCodec::load(const std::string& filename) const {
  size_t fsize = get_filesize(filename);
  fsize -= 8; //remove the first two entries
  // read the first two 4-byte integers in the file, convert to unsigned
  std::ifstream ifile(filename.c_str(), std::ios::binary|std::ios::in);
  if (!ifile) throw io::FileNotReadable(filename);
  uint32_t nsamples, framesize;
  ifile.read((char*)&nsamples, sizeof(uint32_t));
  ifile.read((char*)&framesize, sizeof(uint32_t));
  // are those floats or doubles?
  if (fsize == (nsamples*framesize*sizeof(float))) {
    blitz::Array<float,2> data(nsamples,framesize);
    ifile.read((char*)data.data(), sizeof(float)*framesize*nsamples);
    return io::detail::InlinedArrayImpl(data);
  }
  else if (fsize == (nsamples*framesize*sizeof(double))) {
    // this is the normal case: dealing with single-precision floats
    blitz::Array<double,2> data(nsamples,framesize);
    ifile.read((char*)data.data(), sizeof(double)*framesize*nsamples);
    return io::detail::InlinedArrayImpl(data);
  }
  else {
    ifile.close();
    throw io::TypeError(Torch::core::array::t_unknown, Torch::core::array::t_float32);
  }
}

void io::T3BinaryArrayCodec::save (const std::string& filename,
    const io::detail::InlinedArrayImpl& data) const {
  //can only save uni-dimensional data, so throw if that is not the case
  if (data.getNDim() != 2) throw io::DimensionError(data.getNDim(), 2);

  //can only save float32 or float64, otherwise, throw.
  if ((data.getElementType() != Torch::core::array::t_float32) && 
      (data.getElementType() != Torch::core::array::t_float64)) {
    throw io::UnsupportedTypeError(data.getElementType()); 
  }

  std::ofstream ofile(filename.c_str(), std::ios::binary|std::ios::out);
  const uint32_t nsamples = data.getShape()[0];
  const uint32_t framesize = data.getShape()[1];
  ofile.write((const char*)&nsamples, sizeof(uint32_t));
  ofile.write((const char*)&framesize, sizeof(uint32_t));
  if (data.getElementType() == Torch::core::array::t_float32) {
    blitz::Array<float, 2> save = data.get<float,2>();
    if (!save.isStorageContiguous()) save.reference(save.copy());
    ofile.write((const char*)save.data(), save.extent(0)*save.extent(1)*sizeof(float));
  }
  else { //it is a t_float64
    blitz::Array<double, 2> save = data.get<double,2>();
    if (!save.isStorageContiguous()) save.reference(save.copy());
    ofile.write((const char*)save.data(), save.extent(0)*save.extent(1)*sizeof(double));
  }
  ofile.close();
}
