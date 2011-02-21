/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun 20 Feb 09:55:32 2011 
 *
 * Implements the matlab (.mat) arrayset codec using matio. We do it like this:
 * The first variable to be read in the file defines the type of all the
 * elements that can exist in the file. If we find an element that does not
 * match this type, we raise an exception. Variables *have* to be called
 * "array_%d" or my internal mechanics will not work properly.
 *
 * When we write a file, the arrays in this .mat file will be named like
 * "array_1", "array_2", etc.
 */

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

#include "database/MatArraysetCodec.h"
#include "database/MatUtils.h"
#include "database/ArraysetCodecRegistry.h"
#include "database/Exception.h"

namespace db = Torch::database;

//Takes care of the codec registration.
static bool register_codec() {
  db::ArraysetCodecRegistry::addCodec(boost::shared_ptr<db::ArraysetCodec>(new db::MatArraysetCodec())); 
  return true;
}

static bool codec_registered = register_codec(); 

db::MatArraysetCodec::MatArraysetCodec()
  : m_name("matlab.arrayset.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".mat");
}

db::MatArraysetCodec::~MatArraysetCodec() { }

/**
 * This is a local variant of the peek routine, that also returns the "highest"
 * named "array_%d" object in the file
 */
static size_t max_peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape, size_t& samples) {

  static const boost::regex allowed_varname("^array_(\\d*)$");
  boost::cmatch what;

  mat_t* mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
  if (!mat) throw db::FileNotReadable(filename);
  matvar_t* matvar = Mat_VarReadNext(mat); //gets the first variable

  //we continue reading until we find a variable that matches our naming
  //convention.
  while (!boost::regex_match(matvar->name, what, allowed_varname) && !feof(mat->fp)) {
    Mat_VarFree(matvar);
    matvar = Mat_VarReadNext(mat); //gets the first variable
  }

  if (!what.size()) throw db::Uninitialized();

  //max is set after this variable
  size_t max = boost::lexical_cast<size_t>(what[0]);
 
  //now that we have found a variable under our name convention, fill the array
  //properties taking that variable as basis
  ndim = matvar->rank;
  for (size_t i=0; i<ndim; ++i) shape[i] = matvar->dims[i];
  eltype = db::detail::torch_element_type(matvar->data_type, matvar->isComplex);

  //checks our support and see if we can load this...
  Mat_VarFree(matvar);

  if (ndim > 4) {
    Mat_Close(mat);
    throw db::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY);
  }
  if (eltype == Torch::core::array::t_unknown) {
    Mat_Close(mat);
    throw db::TypeError(eltype, Torch::core::array::t_float32);
  }

  //if we got here, just continue counting the variables inside. we
  //only read their info since that is faster

  samples = 1; //already read 1
  while (!feof(mat->fp)) {
    matvar = Mat_VarReadNextInfo(mat);
    if (boost::regex_match(matvar->name, what, allowed_varname)) {
      size_t v = boost::lexical_cast<size_t>(what[0]);
      if (v > max) max = v;
      ++samples; //a valid variable name was found
    }
    Mat_VarFree(Mat_VarReadNextInfo(mat));
  }
  Mat_Close(mat);
  return max;
}

void db::MatArraysetCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape, size_t& samples) const {
  max_peek(filename, eltype, ndim, shape, samples);
}

#define DIMSWITCH(T) switch(ndim) { \
  case 1: return db::detail::read_arrayset<T,1>(mat); break; \
  case 2: return db::detail::read_arrayset<T,2>(mat); break; \
  case 3: return db::detail::read_arrayset<T,3>(mat); break; \
  case 4: return db::detail::read_arrayset<T,4>(mat); break; \
  default: throw db::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

#define CDIMSWITCH(T,F) switch(ndim) { \
  case 1: return db::detail::read_complex_arrayset<T,F,1>(mat); break; \
  case 2: return db::detail::read_complex_arrayset<T,F,2>(mat); break; \
  case 3: return db::detail::read_complex_arrayset<T,F,3>(mat); break; \
  case 4: return db::detail::read_complex_arrayset<T,F,4>(mat); break; \
  default: throw db::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

db::detail::InlinedArraysetImpl db::MatArraysetCodec::load
(const std::string& filename) const {
  Torch::core::array::ElementType eltype;
  size_t ndim = 0;
  size_t shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
  size_t samples;
  peek(filename, eltype, ndim, shape, samples);

  //we already did this at peek(), so we know it is not going to fail!
  mat_t* mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);

  try {
    switch (eltype) {
      case Torch::core::array::t_int8: 
        DIMSWITCH(int8_t) 
        break;
      case Torch::core::array::t_int16: 
        DIMSWITCH(int16_t) 
        break;
      case Torch::core::array::t_int32: 
        DIMSWITCH(int32_t) 
        break;
      case Torch::core::array::t_int64: 
        DIMSWITCH(int64_t) 
        break;
      case Torch::core::array::t_uint8: 
        DIMSWITCH(uint8_t) 
        break;
      case Torch::core::array::t_uint16: 
        DIMSWITCH(uint16_t) 
        break;
      case Torch::core::array::t_uint32: 
        DIMSWITCH(uint32_t) 
        break;
      case Torch::core::array::t_uint64: 
        DIMSWITCH(uint64_t) 
        break;
      case Torch::core::array::t_float32: 
        DIMSWITCH(float) 
        break;
      case Torch::core::array::t_float64: 
        DIMSWITCH(double) 
        break;
      case Torch::core::array::t_complex64: 
        CDIMSWITCH(std::complex<float>, float) 
        break;
      case Torch::core::array::t_complex128: 
        CDIMSWITCH(std::complex<double>, double) 
        break;
      default:
        throw Torch::core::Exception(); //shut-up gcc
    }
  } catch (Torch::core::Exception& ex) {
    Mat_Close(mat);
    throw;
  }

  Mat_Close(mat);
}

#undef DIMSWITCH
#undef CDIMSWITCH

#define DIMSWITCH(T) switch(ndim) { \
  case 1: return db::detail::read_array<T,1>(mat); break; \
  case 2: return db::detail::read_array<T,2>(mat); break; \
  case 3: return db::detail::read_array<T,3>(mat); break; \
  case 4: return db::detail::read_array<T,4>(mat); break; \
  default: throw db::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

#define CDIMSWITCH(T,F) switch(ndim) { \
  case 1: return db::detail::read_complex_array<T,F,1>(mat); break; \
  case 2: return db::detail::read_complex_array<T,F,2>(mat); break; \
  case 3: return db::detail::read_complex_array<T,F,3>(mat); break; \
  case 4: return db::detail::read_complex_array<T,F,4>(mat); break; \
  default: throw db::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

db::Array db::MatArraysetCodec::load
(const std::string& filename, size_t id) const {
  Torch::core::array::ElementType eltype;
  size_t ndim = 0;
  size_t shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
  size_t samples;
  peek(filename, eltype, ndim, shape, samples);
  if (id > samples) throw db::IndexError(id);

  //we already did this at peek(), so we know it is not going to fail!
  mat_t* mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);

  //we skip to the point we want
  size_t skip = id - 1;
  while (skip--) {
    Mat_VarFree(Mat_VarReadNextInfo(mat));
    ++samples;
  }

  //then we do a normal array readout, as in an ArrayCodec
  try {
    switch (eltype) {
      case Torch::core::array::t_int8: 
        DIMSWITCH(int8_t) 
        break;
      case Torch::core::array::t_int16: 
        DIMSWITCH(int16_t) 
        break;
      case Torch::core::array::t_int32: 
        DIMSWITCH(int32_t) 
        break;
      case Torch::core::array::t_int64: 
        DIMSWITCH(int64_t) 
        break;
      case Torch::core::array::t_uint8: 
        DIMSWITCH(uint8_t) 
        break;
      case Torch::core::array::t_uint16: 
        DIMSWITCH(uint16_t) 
        break;
      case Torch::core::array::t_uint32: 
        DIMSWITCH(uint32_t) 
        break;
      case Torch::core::array::t_uint64: 
        DIMSWITCH(uint64_t) 
        break;
      case Torch::core::array::t_float32: 
        DIMSWITCH(float) 
        break;
      case Torch::core::array::t_float64: 
        DIMSWITCH(double) 
        break;
      case Torch::core::array::t_complex64: 
        CDIMSWITCH(std::complex<float>, float) 
        break;
      case Torch::core::array::t_complex128: 
        CDIMSWITCH(std::complex<double>, double) 
        break;
      default:
        throw Torch::core::Exception(); //shut-up gcc
    }
  } catch (Torch::core::Exception& ex) {
    Mat_Close(mat);
    throw;
  }

  Mat_Close(mat);
}

#undef DIMSWITCH
#undef CDIMSWITCH

#define DIMSWITCH(T) switch(array.getNDim()) { \
  case 1: db::detail::write_array<T,1>(mat, varname, array.get()); break; \
  case 2: db::detail::write_array<T,2>(mat, varname, array.get()); break; \
  case 3: db::detail::write_array<T,3>(mat, varname, array.get()); break; \
  case 4: db::detail::write_array<T,4>(mat, varname, array.get()); break; \
  default: throw db::DimensionError(array.getNDim(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

#define CDIMSWITCH(T,F) switch(array.getNDim()) { \
  case 1: db::detail::write_complex_array<T,F,1>(mat, varname, array.get()); break; \
  case 2: db::detail::write_complex_array<T,F,2>(mat, varname, array.get()); break; \
  case 3: db::detail::write_complex_array<T,F,3>(mat, varname, array.get()); break; \
  case 4: db::detail::write_complex_array<T,F,4>(mat, varname, array.get()); break; \
  default: throw db::DimensionError(array.getNDim(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

void db::MatArraysetCodec::append
(const std::string& filename, const Array& array) const {
  Torch::core::array::ElementType eltype;
  size_t ndim = 0;
  size_t shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
  size_t samples;
  size_t maxnumber = max_peek(filename, eltype, ndim, shape, samples);

  //some checks before letting it append
  if (eltype != array.getElementType()) 
    throw db::TypeError(array.getElementType(), eltype);
  if (ndim != array.getNDim())
    throw db::DimensionError(array.getNDim(), ndim);
  for (size_t i=0; i<ndim; ++i) {
    if (shape[i] != array.getShape()[i])
      throw db::DimensionError(array.getShape()[i], shape[i]);
  }

  //if you get to this point, the array is compatible, let's give it a unique 
  //name based on our max_peek() findings.
  boost::format fmt_varname("array_%d");
  fmt_varname % (maxnumber+1);
  const char* varname = fmt_varname.str().c_str();

  mat_t* mat = Mat_Open(filename.c_str(), MAT_ACC_RDWR);
  if (!mat) throw db::FileNotReadable(filename.c_str());
  try {
    switch(array.getElementType()) {
      case Torch::core::array::t_int8: 
        DIMSWITCH(int8_t) 
        break;
      case Torch::core::array::t_int16: 
        DIMSWITCH(int16_t) 
        break;
      case Torch::core::array::t_int32: 
        DIMSWITCH(int32_t) 
        break;
      case Torch::core::array::t_int64: 
        DIMSWITCH(int64_t) 
        break;
      case Torch::core::array::t_uint8: 
        DIMSWITCH(uint8_t) 
        break;
      case Torch::core::array::t_uint16: 
        DIMSWITCH(uint16_t) 
        break;
      case Torch::core::array::t_uint32: 
        DIMSWITCH(uint32_t) 
        break;
      case Torch::core::array::t_uint64: 
        DIMSWITCH(uint64_t) 
        break;
      case Torch::core::array::t_float32: 
        DIMSWITCH(float) 
        break;
      case Torch::core::array::t_float64: 
        DIMSWITCH(double) 
        break;
      case Torch::core::array::t_complex64: 
        CDIMSWITCH(std::complex<float>, float) 
        break;
      case Torch::core::array::t_complex128: 
        CDIMSWITCH(std::complex<double>, double) 
        break;
      default:
        Mat_Close(mat);
        throw Torch::core::Exception(); //shut-up gcc
    }
  } catch (Torch::core::Exception& ex) {
    Mat_Close(mat);
    throw; //re-throw
  }

  Mat_Close(mat);
}

#undef DIMSWITCH
#undef CDIMSWITCH

#define DIMSWITCH(T) switch(data.getNDim()) { \
  case 1: db::detail::write_arrayset<T,1>(mat, fmt_varname, data); break; \
  case 2: db::detail::write_arrayset<T,2>(mat, fmt_varname, data); break; \
  case 3: db::detail::write_arrayset<T,3>(mat, fmt_varname, data); break; \
  case 4: db::detail::write_arrayset<T,4>(mat, fmt_varname, data); break; \
  default: throw db::DimensionError(data.getNDim(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

#define CDIMSWITCH(T,F) switch(data.getNDim()) { \
  case 1: db::detail::write_complex_arrayset<T,F,1>(mat, fmt_varname, data); break; \
  case 2: db::detail::write_complex_arrayset<T,F,2>(mat, fmt_varname, data); break; \
  case 3: db::detail::write_complex_arrayset<T,F,3>(mat, fmt_varname, data); break; \
  case 4: db::detail::write_complex_arrayset<T,F,4>(mat, fmt_varname, data); break; \
  default: throw db::DimensionError(data.getNDim(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

void db::MatArraysetCodec::save (const std::string& filename, 
    const detail::InlinedArraysetImpl& data) const {
  
  //this file is supposed to hold a single arrayset. delete it if it exists
  boost::filesystem::path path (filename);
  if (boost::filesystem::exists(path)) boost::filesystem::remove(path);

  static boost::format fmt_varname("array_%d");

  mat_t* mat = Mat_Open(filename.c_str(), MAT_ACC_RDWR);
  if (!mat) throw db::FileNotReadable(filename.c_str());

  try {
    switch(data.getElementType()) {
      case Torch::core::array::t_int8: 
        DIMSWITCH(int8_t) 
        break;
      case Torch::core::array::t_int16: 
        DIMSWITCH(int16_t) 
        break;
      case Torch::core::array::t_int32: 
        DIMSWITCH(int32_t) 
        break;
      case Torch::core::array::t_int64: 
        DIMSWITCH(int64_t) 
        break;
      case Torch::core::array::t_uint8: 
        DIMSWITCH(uint8_t) 
        break;
      case Torch::core::array::t_uint16: 
        DIMSWITCH(uint16_t) 
        break;
      case Torch::core::array::t_uint32: 
        DIMSWITCH(uint32_t) 
        break;
      case Torch::core::array::t_uint64: 
        DIMSWITCH(uint64_t) 
        break;
      case Torch::core::array::t_float32: 
        DIMSWITCH(float) 
        break;
      case Torch::core::array::t_float64: 
        DIMSWITCH(double) 
        break;
      case Torch::core::array::t_complex64: 
        CDIMSWITCH(std::complex<float>, float) 
        break;
      case Torch::core::array::t_complex128: 
        CDIMSWITCH(std::complex<double>, double) 
        break;
      default:
        Mat_Close(mat);
        throw Torch::core::Exception(); //shut-up gcc
    }
  } catch (Torch::core::Exception& ex) {
    Mat_Close(mat);
    throw; //re-throw
  }

  Mat_Close(mat);
}
