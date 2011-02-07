/**
 * @file database/src/Array.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the database::Array class
 */

#include "database/Array.h"

namespace db = Torch::database;
namespace core = Torch::core;

db::Array::Array(size_t id=0)
  : m_parent(),
    m_id(id),
    m_is_loaded(False),
    m_filename(""),
    //m_codec(),
    m_elementtype(core::array::t_unknown),
    m_ndim(0),
    m_bzarray(0)
{
}

db::Array::Array(const std::string& filename, const std::string& codecname,
    size_it id=0)
  : m_parent(),
    m_id(id),
    m_is_loaded(False),
    m_filename(filename),
    //m_codec(),
    m_elementtype(core::array::t_unknown),
    m_ndim(0),
    m_bzarray(0)
{
  setFilename(filename, codecname);
}

db::Array::Array(const db::Array& other) 
  : m_parent(),
    m_id(other.m_id),
    m_is_loaded(other.m_is_loaded),
    m_filename(other.m_filename),
    //m_codec(other.m_codec),
    m_elementtype(other.m_elementtype),
    m_ndim(other.m_ndim),
    m_bzarray(other.cloneBlitzArray())
{
}

db::Array::~Array() {
  deleteBlitzArray();
}

db::Array& db::Array::operator= (const db::Array& other) {
  m_parent.reset(); 
  m_id = other.m_id;
  m_is_loaded = other.m_is_loaded;
  m_filename = other.m_filename;
  //m_codec = other.m_codec;
  m_elementtype = other.m_elementtype;
  m_ndim = other.m_ndim;
  m_bzarray = other.cloneBlitzArray();
  return *this;
}

void db::Array::clear() {
  m_is_loaded = false;
  m_filename = "";
  //m_codec.reset();
  deleteBlitzArray();
  m_elementtype = core::array::t_unknown;
  m_ndim = 0;
}

void db::Array::setFilename(const std::string& /*filename*/, 
    const std::string& /*codecname*/) {
  if (m_elementtype != core::array::t_unknown && m_ndim != 0) {
    //already initialized, has to obey current settings
    /**
    //If the named codec does not exist, we raise an exception:
    if (codecname.size()) {
      codec = db::CodecRegistry::getCodecByExtension(m_filename);
    }
    else codec = db::CodecRegistry::getCodecByName(codecname);

    //The codec should tell us the element type and shape of the data
    core::array::ElementType eltype = core::array::t_unknown;
    size_t ndim = 0;
    codec->peek(m_filename, eltype, ndim);
    if (eltype != m_elementtype) throw TypeError();
    if (ndim != m_ndim) throw DimensionError();
    **/
  else {
    //not yet initialized
    /**
    //If the named codec does not exist, we raise an exception:
    if (codecname.size()) {
      codec = db::CodecRegistry::getCodecByExtension(m_filename);
    }
    else codec = db::CodecRegistry::getCodecByName(codecname);

    //The codec should tell us the element type and shape of the data
    codec->peek(m_filename, m_elementtype, m_ndim);
    **/
  }
}

void db::Array::setParent (boost::shared_ptr<Arrayset> parent, size_t id)
  if (m_elementtype != core::array::t_unknown && m_ndim != 0) {
    //already initialized, has to obey current settings
    if (parent->getElementType() != m_elementtype) throw TypeError();
    if (parent->getNDim() != m_ndim) throw DimensionError();
    //if you survived up to this point, just call the parent to remove myself
    //from there and attach to this new parent
    if (m_parent) m_parent->removeArray(*this);
    m_id = id; //reset the id before insertion.
    parent->addArray(*this);
    m_parent = parent;
  else {
    //not yet initialized
    m_parent = parent;
    m_elementtype = parent->getElementType();
    m_ndim = parent->getNDim();
    m_id = id;
    m_parent->addArray(*this);
  }
}

static inline template<typename T, int D> blitz::Array<T,D>* castBzArray
(void* bzarray) {
  return static_cast<blitz::Array<T,D> >(bzarray);
}

static inline template<typename T, int D> deleteBzArray(void* bzarray) {
  delete doCast<T,D>(bzarray);
}

static inline template<typename T, int D> void* cloneBzArray(void* bzarray) {
  return new blitz::Array<T,D>(doCast<T,D>(bzarray)->copy());
}

static inline template<typename T, int D> void* getBzArray(void* bzarray) {
  return new blitz::Array<T,D>(*doCast<T,D>(bzarray));
}

static inline template<typename Text, typename Tint, int D> 
blitz::Array<Text,D> castTypeBzArray(void* bzarray) {
  blitz::Array<Tint,D>* internal = castBzArray<Tint,D>(bzarray);
  //TODO: Ask LES how to use blitz::complex_cast<>() in this situation...
  return blitz::cast<Text>(*internal);
}

#define DIMSWITCH(T,N,D,FUNC) case N: \
  switch(D) { \
  case 1: return FUNC<T,1>(m_bzarray); \
  case 2: return FUNC<T,2>(m_bzarray); \
  case 3: return FUNC<T,3>(m_bzarray); \
  case 4: return FUNC<T,4>(m_bzarray); }

void* db::Array::cloneBlitzArray() const {
  switch(m_elementtype) {
    DIMSWITCH(bool, core::array::t_bool, m_ndim, cloneBzArray)
    DIMSWITCH(int8_t, core::array::t_int8, m_ndim, cloneBzArray)
    DIMSWITCH(int16_t, core::array::t_int16, m_ndim, cloneBzArray)
    DIMSWITCH(int32_t, core::array::t_int32, m_ndim, cloneBzArray)
    DIMSWITCH(int64_t, core::array::t_int64, m_ndim, cloneBzArray)
    DIMSWITCH(uint8_t, core::array::t_uint8, m_ndim, cloneBzArray)
    DIMSWITCH(uint16_t, core::array::t_uint16, m_ndim, cloneBzArray)
    DIMSWITCH(uint32_t, core::array::t_uint32, m_ndim, cloneBzArray)
    DIMSWITCH(uint64_t, core::array::t_uint64, m_ndim, cloneBzArray)
    DIMSWITCH(float, core::array::t_float32, m_ndim, cloneBzArray)
    DIMSWITCH(double, core::array::t_float64, m_ndim, cloneBzArray)
    DIMSWITCH(long double, core::array::t_float128, m_ndim, cloneBzArray)
    DIMSWITCH(std::complex<float>, core::array::t_complex64, m_ndim, cloneBzArray)
    DIMSWITCH(std::complex<double>, core::array::t_complex128, m_ndim, cloneBzArray)
    DIMSWITCH(std::complex<long double>, core::array::t_complex256, m_ndim, cloneBzArray)
    default:
      throw TypeError();
      break;
  }
}

void* db::Array::getBlitzArray() {
  switch(m_elementtype) {
    DIMSWITCH(bool, core::array::t_bool, m_ndim, getBzArray)
    DIMSWITCH(int8_t, core::array::t_int8, m_ndim, getBzArray)
    DIMSWITCH(int16_t, core::array::t_int16, m_ndim, getBzArray)
    DIMSWITCH(int32_t, core::array::t_int32, m_ndim, getBzArray)
    DIMSWITCH(int64_t, core::array::t_int64, m_ndim, getBzArray)
    DIMSWITCH(uint8_t, core::array::t_uint8, m_ndim, getBzArray)
    DIMSWITCH(uint16_t, core::array::t_uint16, m_ndim, getBzArray)
    DIMSWITCH(uint32_t, core::array::t_uint32, m_ndim, getBzArray)
    DIMSWITCH(uint64_t, core::array::t_uint64, m_ndim, getBzArray)
    DIMSWITCH(float, core::array::t_float32, m_ndim, getBzArray)
    DIMSWITCH(double, core::array::t_float64, m_ndim, getBzArray)
    DIMSWITCH(long double, core::array::t_float128, m_ndim, getBzArray)
    DIMSWITCH(std::complex<float>, core::array::t_complex64, m_ndim, getBzArray)
    DIMSWITCH(std::complex<double>, core::array::t_complex128, m_ndim, getBzArray)
    DIMSWITCH(std::complex<long double>, core::array::t_complex256, m_ndim, getBzArray)
    default:
      throw TypeError();
      break;
  }
}

void db::Array::deleteBlitzArray() {
  switch(m_elementtype) {
    DIMSWITCH(bool, core::array::t_bool, m_ndim, deleteBzArray)
    DIMSWITCH(int8_t, core::array::t_int8, m_ndim, deleteBzArray)
    DIMSWITCH(int16_t, core::array::t_int16, m_ndim, deleteBzArray)
    DIMSWITCH(int32_t, core::array::t_int32, m_ndim, deleteBzArray)
    DIMSWITCH(int64_t, core::array::t_int64, m_ndim, deleteBzArray)
    DIMSWITCH(uint8_t, core::array::t_uint8, m_ndim, deleteBzArray)
    DIMSWITCH(uint16_t, core::array::t_uint16, m_ndim, deleteBzArray)
    DIMSWITCH(uint32_t, core::array::t_uint32, m_ndim, deleteBzArray)
    DIMSWITCH(uint64_t, core::array::t_uint64, m_ndim, deleteBzArray)
    DIMSWITCH(float, core::array::t_float32, m_ndim, deleteBzArray)
    DIMSWITCH(double, core::array::t_float64, m_ndim, deleteBzArray)
    DIMSWITCH(long double, core::array::t_float128, m_ndim, deleteBzArray)
    DIMSWITCH(std::complex<float>, core::array::t_complex64, m_ndim, deleteBzArray)
    DIMSWITCH(std::complex<double>, core::array::t_complex128, m_ndim, deleteBzArray)
    DIMSWITCH(std::complex<long double>, core::array::t_complex256, m_ndim, deleteBzArray)
    default:
      throw TypeError();
      break;
  }
}
