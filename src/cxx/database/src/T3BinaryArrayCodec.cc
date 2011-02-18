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

#include "database/T3BinaryArrayCodec.h"
#include "database/ArrayCodecRegistry.h"
#include "database/Exception.h"
#include <fstream>

//some infrastructure to check the file size
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace db = Torch::database;

static inline size_t get_filesize(const std::string& filename) {
  struct stat filestatus;
  stat(filename.c_str(), &filestatus);
  return filestatus.st_size;
}

//Takes care of the codec registration.
static bool register_codec() {
  db::ArrayCodecRegistry::addCodec(boost::shared_ptr<db::ArrayCodec>(new db::T3BinaryArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

db::T3BinaryArrayCodec::T3BinaryArrayCodec(bool save_in_float32)
  : m_name("torch3.array.binary"),
    m_extensions(),
    m_float(save_in_float32)
{ 
  m_extensions.push_back(".bindata");
}

db::T3BinaryArrayCodec::~T3BinaryArrayCodec() { }

void db::T3BinaryArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const {
  size_t fsize = get_filesize(filename);
  fsize -= 8; // remove the first two entries
  // read the first two 4-byte integers in the file, convert to unsigned
  std::ifstream ifile(filename.c_str(), std::ios::binary|std::ios::in);
  if (!ifile) throw db::FileNotReadable(filename);
  uint32_t nsamples, framesize;
  ifile >> nsamples;
  ifile >> framesize;
  ifile.close();
  // are those floats or doubles?
  if (fsize == (nsamples*framesize)) eltype = Torch::core::array::t_float32;
  else if (fsize == (2*nsamples*framesize)) eltype = Torch::core::array::t_float64;
  else throw db::TypeError(Torch::core::array::t_float32, Torch::core::array::t_unknown);
  ndim = 1;
  shape[0] = framesize;
}

db::detail::InlinedArrayImpl 
db::T3BinaryArrayCodec::load(const std::string& filename) const {
  size_t fsize = get_filesize(filename);
  fsize -= 8; //remove the first two entries
  // read the first two 4-byte integers in the file, convert to unsigned
  std::ifstream ifile(filename.c_str(), std::ios::binary|std::ios::in);
  if (!ifile) throw db::FileNotReadable(filename);
  uint32_t nsamples, framesize;
  ifile >> nsamples;
  ifile >> framesize;
  // are those floats or doubles?
  if (fsize == (nsamples*framesize)) {
    blitz::Array<float,1> data(framesize);
    for (size_t i=0; i<framesize; ++i) ifile >> data(i);
    return db::detail::InlinedArrayImpl(data);
  }
  else if (fsize == (2*nsamples*framesize)) {
    // this is the normal case: dealing with single-precision floats
    blitz::Array<double,1> data(framesize);
    for (size_t i=0; i<framesize; ++i) ifile >> data(i);
    return db::detail::InlinedArrayImpl(data);
  }
  throw db::TypeError(Torch::core::array::t_float32, Torch::core::array::t_unknown);
}

void db::T3BinaryArrayCodec::save (const std::string& filename,
    const db::detail::InlinedArrayImpl& data) const {
  //can only save uni-dimensional data, so throw if that is not the case
  if (data.getNDim() != 1) throw db::DimensionError(data.getNDim(), 1);

  std::ofstream ofile(filename.c_str(), std::ios::binary|std::ios::out);
  const uint32_t nsamples = 1;
  ofile << nsamples;
  const uint32_t length = data.getShape()[0];
  ofile << length;
  if (m_float) {
    blitz::Array<float, 1> save = data.cast<float,1>();
    for (blitz::sizeType i=0; i<save.extent(0); ++i) ofile << save(i); 
  }
  else {
    blitz::Array<double, 1> save = data.cast<double,1>();
    for (blitz::sizeType i=0; i<save.extent(0); ++i) ofile << save(i); 
  }
  ofile.close();
}
