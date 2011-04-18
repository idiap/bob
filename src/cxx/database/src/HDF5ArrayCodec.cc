/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Sun 17 Apr 11:24:32 2011 CEST
 *
 * @brief Implements the HDF5 (.hdf5) array codec
 */

#include <boost/shared_array.hpp>
#include <boost/filesystem.hpp>

#include "database/HDF5ArrayCodec.h"
#include "database/HDF5File.h"
#include "database/ArrayCodecRegistry.h"
#include "database/HDF5Exception.h"

namespace db = Torch::database;

//Takes care of the codec registration.
static bool register_codec() {
  db::ArrayCodecRegistry::addCodec(boost::shared_ptr<db::ArrayCodec>(new db::HDF5ArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();
static boost::shared_ptr<db::HDF5Error> init = db::HDF5Error::instance();

db::HDF5ArrayCodec::HDF5ArrayCodec()
  : m_name("hdf5.array.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".hdf5");
}

db::HDF5ArrayCodec::~HDF5ArrayCodec() { }

void db::HDF5ArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const {
  db::HDF5File f(filename, db::HDF5File::in);
  std::vector<std::string> paths;
  f.paths(paths);
  if (!paths.size()) throw db::HDF5InvalidPath(filename, "array");
  const std::string& name = paths[0];
  const db::HDF5Type& descr = f.describe(name);
  eltype = descr.element_type(); 
  if (eltype == Torch::core::array::t_unknown) {
    throw db::UnsupportedTypeError(eltype);
  }
  ndim = descr.shape().n();
  if (ndim > 4) {
    throw db::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY);
  }
  for (size_t i=0; i<ndim; ++i) shape[i] = descr.shape()[i];
}

template <typename T, int N>
static db::detail::InlinedArrayImpl read_array (db::HDF5File& f,
    const std::string& path) {
  const db::HDF5Type& descr = f.describe(path);
  blitz::TinyVector<int,N> shape;
  descr.shape().set(shape);
  blitz::Array<T,N> retval(shape);
  f.readArray(path, 0, retval);
  return db::detail::InlinedArrayImpl(retval);
}

#define DIMSWITCH(T) switch(descr.shape().n()) { \
  case 1: return read_array<T,1>(f, name); break; \
  case 2: return read_array<T,2>(f, name); break; \
  case 3: return read_array<T,3>(f, name); break; \
  case 4: return read_array<T,4>(f, name); break; \
  default: throw db::DimensionError(descr.shape().n(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

db::detail::InlinedArrayImpl 
db::HDF5ArrayCodec::load(const std::string& filename) const {
  db::HDF5File f(filename, db::HDF5File::in);
  std::vector<std::string> paths;
  f.paths(paths);
  if (!paths.size()) throw db::HDF5InvalidPath(filename, "array");
  const std::string& name = paths[0];
  const db::HDF5Type& descr = f.describe(name);
  switch (descr.element_type()) {
    case Torch::core::array::t_bool:
      DIMSWITCH(bool) 
        break;
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
    case Torch::core::array::t_float128: 
      DIMSWITCH(long double) 
        break;
    case Torch::core::array::t_complex64: 
      DIMSWITCH(std::complex<float>) 
        break;
    case Torch::core::array::t_complex128: 
      DIMSWITCH(std::complex<double>)
        break;
    case Torch::core::array::t_complex256: 
      DIMSWITCH(std::complex<long double>)
        break;
    default:
      break;
  }
  throw Torch::database::UnsupportedTypeError(descr.element_type());
}

#undef DIMSWITCH

#define DIMSWITCH(T) switch(data.getNDim()) { \
  case 1: f.appendArray(varname, data.get<T,1>()); break; \
  case 2: f.appendArray(varname, data.get<T,2>()); break; \
  case 3: f.appendArray(varname, data.get<T,3>()); break; \
  case 4: f.appendArray(varname, data.get<T,4>()); break; \
  default: throw db::DimensionError(data.getNDim(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

void db::HDF5ArrayCodec::save (const std::string& filename,
    const db::detail::InlinedArrayImpl& data) const {
  static std::string varname("array");
  db::HDF5File f(filename, db::HDF5File::trunc);
  switch(data.getElementType()) {
    case Torch::core::array::t_bool: 
      DIMSWITCH(bool) 
        break;
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
    case Torch::core::array::t_float128: 
      DIMSWITCH(long double) 
        break;
    case Torch::core::array::t_complex64: 
      DIMSWITCH(std::complex<float>)
        break;
    case Torch::core::array::t_complex128: 
      DIMSWITCH(std::complex<double>)
        break;
    case Torch::core::array::t_complex256: 
      DIMSWITCH(std::complex<long double>)
        break;
    default:
      throw Torch::database::UnsupportedTypeError(data.getElementType());
  }
}

#undef DIMSWITCH
