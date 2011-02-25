/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun 20 Feb 09:51:30 2011 
 *
 * @brief Implements the matlab (.mat) array codec using matio
 */

#include <boost/shared_array.hpp>
#include <boost/filesystem.hpp>

#include "database/MatArrayCodec.h"
#include "database/MatUtils.h"
#include "database/ArrayCodecRegistry.h"
#include "database/Exception.h"

namespace db = Torch::database;

//Takes care of the codec registration.
static bool register_codec() {
  db::ArrayCodecRegistry::addCodec(boost::shared_ptr<db::ArrayCodec>(new db::MatArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

db::MatArrayCodec::MatArrayCodec()
  : m_name("matlab.array.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".mat");
}

db::MatArrayCodec::~MatArrayCodec() { }

void db::MatArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const {

  boost::shared_ptr<mat_t> mat = db::detail::make_matfile(filename, MAT_ACC_RDWR);
  if (!mat) throw db::FileNotReadable(filename);

  boost::shared_ptr<matvar_t> matvar = db::detail::make_matvar(mat); 

  ndim = matvar->rank;
  for (size_t i=0; i<ndim; ++i) shape[i] = matvar->dims[i];
  eltype = db::detail::torch_element_type(matvar->data_type, matvar->isComplex);

  if (ndim > 4) {
    throw db::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY);
  }
  if (eltype == Torch::core::array::t_unknown) {
    throw db::UnsupportedTypeError(eltype);
  }
}

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

db::detail::InlinedArrayImpl 
db::MatArrayCodec::load(const std::string& filename) const {
  Torch::core::array::ElementType eltype;
  size_t ndim = 0;
  size_t shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
  peek(filename, eltype, ndim, shape);

  //we already did this at peek(), so we know it is not going to fail!
  boost::shared_ptr<mat_t> mat = db::detail::make_matfile(filename, MAT_ACC_RDWR);

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

#define DIMSWITCH(T) switch(data.getNDim()) { \
  case 1: db::detail::write_array<T,1>(mat, varname, data); break; \
  case 2: db::detail::write_array<T,2>(mat, varname, data); break; \
  case 3: db::detail::write_array<T,3>(mat, varname, data); break; \
  case 4: db::detail::write_array<T,4>(mat, varname, data); break; \
  default: throw db::DimensionError(data.getNDim(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

#define CDIMSWITCH(T,F) switch(data.getNDim()) { \
  case 1: db::detail::write_complex_array<T,F,1>(mat, varname, data); break; \
  case 2: db::detail::write_complex_array<T,F,2>(mat, varname, data); break; \
  case 3: db::detail::write_complex_array<T,F,3>(mat, varname, data); break; \
  case 4: db::detail::write_complex_array<T,F,4>(mat, varname, data); break; \
  default: throw db::DimensionError(data.getNDim(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

void db::MatArrayCodec::save (const std::string& filename,
    const db::detail::InlinedArrayImpl& data) const {
  static std::string varname("array");

  //this file is supposed to hold a single array. delete it if it exists
  boost::filesystem::path path (filename);
  if (boost::filesystem::exists(path)) boost::filesystem::remove(path);

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

#undef DIMSWITCH
#undef CDIMSWITCH
