/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Sun 17 Apr 12:05:21 2011 CEST
 *
 * Implements the HDF5 (.hdf5) arrayset codec.
 */

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "io/HDF5ArraysetCodec.h"
#include "io/HDF5File.h"
#include "io/ArraysetCodecRegistry.h"
#include "io/HDF5Exception.h"

namespace io = Torch::io;

//Takes care of the codec registration.
static bool register_codec() {
  io::ArraysetCodecRegistry::addCodec(boost::shared_ptr<io::ArraysetCodec>(new io::HDF5ArraysetCodec())); 
  return true;
}

static bool codec_registered = register_codec(); 

io::HDF5ArraysetCodec::HDF5ArraysetCodec()
  : m_name("hdf5.arrayset.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".h5");
  m_extensions.push_back(".hdf5");
}

io::HDF5ArraysetCodec::~HDF5ArraysetCodec() { }

void io::HDF5ArraysetCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape, size_t& samples) const {
  io::HDF5File f(filename, io::HDF5File::in);
  std::vector<std::string> paths;
  f.paths(paths);
  if (!paths.size()) throw io::HDF5InvalidPath(filename, "arrayset");
  const std::string& name = paths[0];
  io::HDF5File::description_t defdescr;
  f.describe(name, defdescr);
  io::HDF5Type& descr = boost::get<0>(defdescr);
  eltype = descr.element_type(); 
  if (eltype == Torch::core::array::t_unknown) {
    throw io::UnsupportedTypeError(eltype);
  }
  ndim = descr.shape().n();
  if (ndim > 4) {
    throw io::DimensionError(ndim, Torch::core::array::N_MAX_DIMENSIONS_ARRAY);
  }
  for (size_t i=0; i<ndim; ++i) shape[i] = descr.shape()[i];
  samples = boost::get<1>(defdescr);
}

template <typename T, int N>
static io::detail::InlinedArraysetImpl read_arrayset (io::HDF5File& f,
    const std::string& path) {
  io::HDF5File::description_t defdescr;
  f.describe(path, defdescr);
  io::HDF5Type& descr = boost::get<0>(defdescr);
  blitz::TinyVector<int,N> shape;
  descr.shape().set(shape);
  io::detail::InlinedArraysetImpl retval;
  for (size_t i=0; i<boost::get<1>(defdescr); ++i) {
    blitz::Array<T,N> tmp(shape);
    f.readArray(path, i, tmp);
    retval.add(io::Array(tmp));
  }
  return retval;
}

#define DIMSWITCH(T) switch(descr.shape().n()) { \
  case 1: return read_arrayset<T,1>(f, name); break; \
  case 2: return read_arrayset<T,2>(f, name); break; \
  case 3: return read_arrayset<T,3>(f, name); break; \
  case 4: return read_arrayset<T,4>(f, name); break; \
  default: throw io::DimensionError(descr.shape().n(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

io::detail::InlinedArraysetImpl io::HDF5ArraysetCodec::load
(const std::string& filename) const {
  io::HDF5File f(filename, io::HDF5File::in);
  std::vector<std::string> paths;
  f.paths(paths);
  if (!paths.size()) throw io::HDF5InvalidPath(filename, "/arrayset");
  const std::string& name = paths[0];
  io::HDF5File::description_t defdescr;
  f.describe(name, defdescr);
  io::HDF5Type& descr = boost::get<0>(defdescr);
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
  throw Torch::io::UnsupportedTypeError(descr.element_type());
}

#undef DIMSWITCH

template <typename T, int N>
static io::Array read_array (io::HDF5File& f, const std::string& path, 
    size_t pos) {
  io::HDF5File::description_t defdescr;
  f.describe(path, defdescr);
  io::HDF5Type& descr = boost::get<0>(defdescr);
  blitz::TinyVector<int,N> shape;
  descr.shape().set(shape);
  blitz::Array<T,N> retval(shape);
  f.readArray(path, pos, retval);
  return io::detail::InlinedArrayImpl(retval);
}

#define DIMSWITCH(T) switch(descr.shape().n()) { \
  case 1: return read_array<T,1>(f, name, id); break; \
  case 2: return read_array<T,2>(f, name, id); break; \
  case 3: return read_array<T,3>(f, name, id); break; \
  case 4: return read_array<T,4>(f, name, id); break; \
  default: throw io::DimensionError(descr.shape().n(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

io::Array io::HDF5ArraysetCodec::load
(const std::string& filename, size_t id) const {
  io::HDF5File f(filename, io::HDF5File::in);
  std::vector<std::string> paths;
  f.paths(paths);
  if (!paths.size()) throw io::HDF5InvalidPath(filename, "/arrayset");
  const std::string& name = paths[0];
  io::HDF5File::description_t defdescr;
  f.describe(name, defdescr);
  if (id >= boost::get<1>(defdescr)) throw io::IndexError(id);
  io::HDF5Type& descr = boost::get<0>(defdescr);
  //then we do a normal array readout, as in an ArrayCodec.load()
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
  throw Torch::io::UnsupportedTypeError(descr.element_type());
}

#undef DIMSWITCH

#define DIMSWITCH(T) switch(array.getNDim()) { \
  case 1: f.appendArray(varname, array.get<T,1>()); break; \
  case 2: f.appendArray(varname, array.get<T,2>()); break; \
  case 3: f.appendArray(varname, array.get<T,3>()); break; \
  case 4: f.appendArray(varname, array.get<T,4>()); break; \
  default: throw io::DimensionError(array.getNDim(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

void io::HDF5ArraysetCodec::append
(const std::string& filename, const Array& array) const {
  static std::string varname("arrayset");
  io::HDF5File f(filename, io::HDF5File::inout);
  switch(array.getElementType()) {
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
      throw Torch::io::UnsupportedTypeError(array.getElementType());
  }
}

#undef DIMSWITCH

template <typename T, int N>
static void add_arrayset(io::HDF5File& f, const std::string& path, 
    const io::detail::InlinedArraysetImpl& data) {
  for (size_t i=0; i<data.size(); ++i) {
    f.appendArray(path, data[i].get().get<T,N>());
  }
}

#define DIMSWITCH(T) switch(data.getNDim()) { \
  case 1: add_arrayset<T,1>(f, varname, data); break; \
  case 2: add_arrayset<T,2>(f, varname, data); break; \
  case 3: add_arrayset<T,3>(f, varname, data); break; \
  case 4: add_arrayset<T,4>(f, varname, data); break; \
  default: throw io::DimensionError(data.getNDim(), Torch::core::array::N_MAX_DIMENSIONS_ARRAY); \
}

void io::HDF5ArraysetCodec::save (const std::string& filename, 
    const io::detail::InlinedArraysetImpl& data) const {
  static std::string varname("arrayset");
  io::HDF5File f(filename, io::HDF5File::trunc);
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
      DIMSWITCH(std::complex<double>)
        break;
    default:
      throw Torch::io::UnsupportedTypeError(data.getElementType());
  }
}

#undef DIMSWITCH
