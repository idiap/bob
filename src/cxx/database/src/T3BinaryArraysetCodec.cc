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

#include "database/T3BinaryArraysetCodec.h"
#include "database/ArraysetCodecRegistry.h"
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
  db::ArraysetCodecRegistry::addCodec(boost::shared_ptr<db::ArraysetCodec>(new db::T3BinaryArraysetCodec())); 
  return true;
}

static bool codec_registered = register_codec(); 

db::T3BinaryArraysetCodec::T3BinaryArraysetCodec(bool save_in_float32)
  : m_name("torch3.arrayset.binary"),
    m_extensions(),
    m_float(save_in_float32)
{ 
  m_extensions.push_back(".bindata");
}

db::T3BinaryArraysetCodec::~T3BinaryArraysetCodec() { }

void db::T3BinaryArraysetCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape, size_t& samples) const {
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
  samples = nsamples;
}

db::detail::InlinedArraysetImpl db::T3BinaryArraysetCodec::load
(const std::string& filename) const {
  size_t fsize = get_filesize(filename);
  fsize -= 8; // remove the first two entries
  // read the first two 4-byte integers in the file, convert to unsigned
  std::ifstream ifile(filename.c_str(), std::ios::binary|std::ios::in);
  if (!ifile) throw db::FileNotReadable(filename);
  uint32_t xsamples, framesize;
  ifile >> xsamples;
  ifile >> framesize;
  // are those floats or doubles?
  db::detail::InlinedArraysetImpl retval;
  if (fsize == (xsamples*framesize)) { //floats of 32-bits
    for (size_t ex=0; ex<xsamples; ++ex) {
      blitz::Array<float, 1> data(framesize);
      for (size_t i=0; i<framesize; ++i) ifile >> data(i);
      retval.add(db::Array(data));
    }
  }
  else if (fsize == (2*xsamples*framesize)) {
    for (size_t ex=0; ex<xsamples; ++ex) {
      blitz::Array<double, 1> data(framesize);
      for (size_t i=0; i<framesize; ++i) ifile >> data(i);
      retval.add(db::Array(data));
    }
  }
  else {
    ifile.close();
    throw db::TypeError(Torch::core::array::t_float32, Torch::core::array::t_unknown);
  }
  ifile.close();
  return retval;
}

db::Array db::T3BinaryArraysetCodec::load
(const std::string& filename, size_t id) const {
  size_t fsize = get_filesize(filename);
  fsize -= 8; // remove the first two entries
  // read the first two 4-byte integers in the file, convert to unsigned
  std::ifstream ifile(filename.c_str(), std::ios::binary|std::ios::in);
  if (!ifile) throw db::FileNotReadable(filename);
  uint32_t xsamples, framesize;
  ifile >> xsamples;
  if (id > xsamples) throw db::IndexError(id);
  ifile >> framesize;
  // are those floats or doubles?
  if (fsize == (xsamples*framesize)) { //floats of 32-bits
    ifile.seekg(8 + (sizeof(float)*(id-1)*framesize)); //move into position
    blitz::Array<float, 1> data(framesize);
    for (size_t i=0; i<framesize; ++i) ifile >> data(i);
    return db::Array(data);
  }
  else if (fsize == (2*xsamples*framesize)) {
    ifile.seekg(8 + (sizeof(double)*(id-1)*framesize)); //move into position
    blitz::Array<double, 1> data(framesize);
    for (size_t i=0; i<framesize; ++i) ifile >> data(i);
    return db::Array(data);
  }
  else {
    ifile.close();
    throw db::TypeError(Torch::core::array::t_float32, Torch::core::array::t_unknown);
  }
}

void db::T3BinaryArraysetCodec::append
(const std::string& filename, const Array& array) const {
  //peek the data to see we are ok
  Torch::core::array::ElementType eltype;
  size_t ndim;
  size_t shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
  size_t samples;
  peek(filename, eltype, ndim, shape, samples);

  //throw if typing is different (remember we only accept 1-D input!!)
  if (array.getNDim() != ndim) throw db::DimensionError(array.getNDim(), ndim);
  if (shape[0] != array.getShape()[0]) throw db::DimensionError(array.getShape()[0], shape[0]);
  if (array.getElementType() != eltype) throw db::TypeError(array.getElementType(), eltype);

  std::ofstream ofile(filename.c_str(), std::ios::binary|std::ios::app);
  if (m_float) {
    blitz::Array<float, 1> save = array.cast<float,1>();
    for (blitz::sizeType i=0; i<save.extent(0); ++i) ofile << save(i); 
  }
  else {
    blitz::Array<double, 1> save = array.cast<double,1>();
    for (blitz::sizeType i=0; i<save.extent(0); ++i) ofile << save(i); 
  }
  ofile.close();
}

void db::T3BinaryArraysetCodec::save (const std::string& filename, 
    const detail::InlinedArraysetImpl& data) const {
  std::ofstream ofile(filename.c_str(), std::ios::binary|std::ios::out);
  uint32_t samples = data.getNSamples();
  uint32_t dim0 = data.getShape()[0];
  ofile << samples << dim0;
  if (m_float) {
    for (size_t i=0; i<data.getNSamples(); ++i) {
      blitz::Array<float, 1> save = data[i+1].cast<float,1>();
      for (blitz::sizeType i=0; i<save.extent(0); ++i) ofile << save(i); 
    }
  }
  else {
    for (size_t i=0; i<data.getNSamples(); ++i) {
      blitz::Array<double, 1> save = data[i+1].cast<double,1>();
      for (blitz::sizeType i=0; i<save.extent(0); ++i) ofile << save(i); 
    }
  }
  ofile.close();
}
