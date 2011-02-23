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

void db::MatArraysetCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape, size_t& samples) const {
  boost::shared_ptr<std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> > > dict = db::detail::list_variables(filename);
  const db::ArrayTypeInfo& info = dict->begin()->second.second;
  eltype = info.eltype;
  ndim = info.ndim;
  for (size_t i=0; i<ndim; ++i) shape[i] = info.shape[i];
  samples = dict->size();
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
  boost::shared_ptr<mat_t> mat = db::detail::make_matfile(filename, MAT_ACC_RDONLY);

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
      break;
  }
  throw Torch::database::UnsupportedTypeError(eltype);
}

#undef DIMSWITCH
#undef CDIMSWITCH

#define DIMSWITCH(T) switch(ndim) { \
  case 1: return db::detail::read_array<T,1>(mat,varname); break; \
  case 2: return db::detail::read_array<T,2>(mat,varname); break; \
  case 3: return db::detail::read_array<T,3>(mat,varname); break; \
  case 4: return db::detail::read_array<T,4>(mat,varname); break; \
  default: throw db::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

#define CDIMSWITCH(T,F) switch(ndim) { \
  case 1: return db::detail::read_complex_array<T,F,1>(mat,varname); break; \
  case 2: return db::detail::read_complex_array<T,F,2>(mat,varname); break; \
  case 3: return db::detail::read_complex_array<T,F,3>(mat,varname); break; \
  case 4: return db::detail::read_complex_array<T,F,4>(mat,varname); break; \
  default: throw db::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

db::Array db::MatArraysetCodec::load
(const std::string& filename, size_t id) const {
  
  boost::shared_ptr<std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> > > dict = db::detail::list_variables(filename);

  if (id > dict->size()) throw db::IndexError(id);

  //the name of interest
  std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> >::iterator 
    it = dict->begin();
  std::advance(it, id-1); //it is now pointing to the correct name to look for.

  //we already did this at peek(), so we know it is not going to fail!
  boost::shared_ptr<mat_t> mat = db::detail::make_matfile(filename, MAT_ACC_RDONLY);
  const std::string& varname = it->second.first;
  Torch::core::array::ElementType eltype = it->second.second.eltype;
  size_t ndim = it->second.second.ndim;

  //then we do a normal array readout, as in an ArrayCodec.load()
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
      break;
  }
  throw Torch::database::UnsupportedTypeError(eltype);
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
  
  size_t next_index = 1; //to be used for the array name "array_%d", bellow

  if (boost::filesystem::exists(filename)) { //the new array must conform!

    boost::shared_ptr<std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> > > dict = db::detail::list_variables(filename);

    //the first entry
    std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> >::iterator 
      it = dict->begin();
    const db::ArrayTypeInfo& info = it->second.second;

    //some checks before letting it append
    if (info.eltype != array.getElementType()) 
      throw db::TypeError(array.getElementType(), info.eltype);
    if (info.ndim != array.getNDim())
      throw db::DimensionError(array.getNDim(), info.ndim);
    for (size_t i=0; i<info.ndim; ++i) {
      if (info.shape[i] != array.getShape()[i])
        throw db::DimensionError(array.getShape()[i], info.shape[i]);
    }

    //max number => last entry
    next_index = dict->rbegin()->first + 1;
  }

  //if you get to this point, the array is compatible, let's give it a unique 
  //name based on our max_peek() findings.
  boost::format fmt_varname("array_%d");
  fmt_varname % next_index;
  const std::string& varname = fmt_varname.str();

  boost::shared_ptr<mat_t> mat = db::detail::make_matfile(filename, MAT_ACC_RDWR);
  if (!mat) throw db::FileNotReadable(filename);

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
      throw Torch::database::UnsupportedTypeError(array.getElementType());
  }
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

  boost::shared_ptr<mat_t> mat = db::detail::make_matfile(filename, MAT_ACC_RDWR);
  if (!mat) throw db::FileNotReadable(filename);

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
      throw Torch::database::UnsupportedTypeError(data.getElementType());
  }
}
