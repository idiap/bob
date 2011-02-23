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
#include <boost/filesystem.hpp>

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

db::T3BinaryArraysetCodec::T3BinaryArraysetCodec()
  : m_name("torch3.arrayset.binary"),
    m_extensions()
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
  ifile.read((char*)&nsamples, sizeof(uint32_t));
  ifile.read((char*)&framesize, sizeof(uint32_t));
  ifile.close();
  // are those floats or doubles?
  if (fsize == (nsamples*framesize*sizeof(float))) eltype = Torch::core::array::t_float32;
  else if (fsize == (nsamples*framesize*sizeof(double))) eltype = Torch::core::array::t_float64;
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
  ifile.read((char*)&xsamples, sizeof(uint32_t));
  ifile.read((char*)&framesize, sizeof(uint32_t));
  // are those floats or doubles?
  db::detail::InlinedArraysetImpl retval;
  if (fsize == (xsamples*framesize*sizeof(float))) { //floats of 32-bits
    for (size_t ex=0; ex<xsamples; ++ex) {
      blitz::Array<float, 1> data(framesize);
      ifile.read((char*)data.data(), sizeof(float)*framesize);
      retval.add(db::Array(data));
    }
  }
  else if (fsize == (xsamples*framesize*sizeof(double))) {
    for (size_t ex=0; ex<xsamples; ++ex) {
      blitz::Array<double, 1> data(framesize);
      ifile.read((char*)data.data(), sizeof(double)*framesize);
      retval.add(db::Array(data));
    }
  }
  else {
    ifile.close();
    throw db::TypeError(Torch::core::array::t_unknown, Torch::core::array::t_float32);
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
  ifile.read((char*)&xsamples, sizeof(uint32_t));
  ifile.read((char*)&framesize, sizeof(uint32_t));
  // are those floats or doubles?
  if (fsize == (xsamples*framesize*sizeof(float))) { //floats of 32-bits
    ifile.seekg(8 + (sizeof(float)*(id-1)*framesize)); //move into position
    blitz::Array<float, 1> data(framesize);
    ifile.read((char*)data.data(), sizeof(float)*framesize);
    return db::Array(data);
  }
  else if (fsize == (xsamples*framesize*sizeof(double))) { //floats of 64-bits
    ifile.seekg(8 + (sizeof(double)*(id-1)*framesize)); //move into position
    blitz::Array<double, 1> data(framesize);
    ifile.read((char*)data.data(), sizeof(double)*framesize);
    return db::Array(data);
  }
  else {
    ifile.close();
    throw db::TypeError(Torch::core::array::t_float32, Torch::core::array::t_unknown);
  }
}

void db::T3BinaryArraysetCodec::append
(const std::string& filename, const Array& array) const {

  std::ofstream ofile;

  if (boost::filesystem::exists(filename)) { //the new array must conform!
    //peek the data to see we are ok by looking existing specifications on file
    Torch::core::array::ElementType eltype;
    size_t ndim;
    size_t shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
    size_t samples;
    peek(filename, eltype, ndim, shape, samples);

    //throw if typing is different (remember we only accept 1-D input!!)
    if (array.getNDim() != ndim) throw db::DimensionError(array.getNDim(), ndim);
    if (shape[0] != array.getShape()[0]) throw db::DimensionError(array.getShape()[0], shape[0]);
    if (array.getElementType() != eltype) throw db::TypeError(array.getElementType(), eltype);

    //and we open it to append - we need to re-write a bit of the header. NB:
    //if you open the file with only std::ios::out, it will truncate it; we use
    //both std::ios::out and std::ios::in to achieve the desired effect
    ofile.open(filename.c_str(), std::ios::binary|std::ios::in|std::ios::out);
    uint32_t nsamples = samples + 1;
    ofile.write((const char*)&nsamples, sizeof(uint32_t));
    ofile.close();
    ofile.open(filename.c_str(), std::ios::binary|std::ios::app);
  }
  else {
    //if the file does not exist, we must start it...
    ofile.open(filename.c_str(), std::ios::binary|std::ios::out);
    //and write some of the variables we need to
    uint32_t nsamples = 1;
    uint32_t framesize = array.getShape()[0];
    ofile.write((const char*)&nsamples, sizeof(uint32_t));
    ofile.write((const char*)&framesize, sizeof(uint32_t));
  }

  //now we let it append
  if (array.getElementType() == Torch::core::array::t_float32) {
    blitz::Array<float, 1> save = array.get<float,1>();
    ofile.write((const char*)save.data(), save.extent(0)*sizeof(float));
  }
  else { //must be double
    blitz::Array<double, 1> save = array.get<double,1>();
    ofile.write((const char*)save.data(), save.extent(0)*sizeof(double));
  }
  ofile.close();
}

void db::T3BinaryArraysetCodec::save (const std::string& filename, 
    const detail::InlinedArraysetImpl& data) const {
  //can only save uni-dimensional data, so throw if that is not the case
  if (data.getNDim() != 1) throw db::DimensionError(data.getNDim(), 1);

  //can only save float32 or float64, otherwise, throw.
  if ((data.getElementType() != Torch::core::array::t_float32) && 
      (data.getElementType() != Torch::core::array::t_float64)) {
    throw db::UnsupportedTypeError(data.getElementType());
  }

  std::ofstream ofile(filename.c_str(), std::ios::binary|std::ios::out);
  uint32_t nsamples = data.getNSamples();
  uint32_t framesize = data.getShape()[0];
  ofile.write((const char*)&nsamples, sizeof(uint32_t));
  ofile.write((const char*)&framesize, sizeof(uint32_t));
  if (data.getElementType() == Torch::core::array::t_float32) {
    for (size_t i=0; i<data.getNSamples(); ++i) {
      blitz::Array<float, 1> save = data[i+1].get<float,1>();
      ofile.write((const char*)save.data(), save.extent(0)*sizeof(float));
    }
  }
  else { //must be double
    for (size_t i=0; i<data.getNSamples(); ++i) {
      blitz::Array<double, 1> save = data[i+1].get<double,1>();
      ofile.write((const char*)save.data(), save.extent(0)*sizeof(double));
    }
  }
  ofile.close();
}
