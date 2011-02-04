/**
 * @file database/src/Array.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the database::Array class
 */

#include "database/Array.h"

namespace db = Torch::database;
namespace core = Torch::core;

/**
 * Attributes
 */
boost::weak_ptr<Arrayset> m_parent_arrayset;
size_t m_id;
bool m_is_loaded;
std::string m_filename;
std::string m_codecname;
void* m_storage;

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
  : m_parent(other.m_parent),
    m_id(other.m_id),
    m_is_loaded(other.m_is_loaded),
    m_filename(other.m_filename),
    //m_codec(other.m_codec),
    m_elementtype(other.m_elementtype),
    m_ndim(other.m_ndim),
    m_bzarray(other.getBlitzArray())
{
}

db::Array::~Array() {
  deleteBlitzArray();
}

db::Array& db::Array::operator= (const db::Array& other) {
  m_parent = other.m_parent;
  m_id = other.m_id;
  m_is_loaded = other.m_is_loaded;
  m_filename = other.m_filename;
  //m_codec = other.m_codec;
  m_elementtype = other.m_elementtype;
  m_ndim = other.m_ndim;
  m_bzarray = other.getBlitzArray();
  return *this;
}

void db::Array::clear() {
  m_is_loaded = false;
  m_filename = "";
  //m_codec.reset();
  m_elementtype = core::array::t_unknown;
  m_ndim = 0;
  deleteBlitzArray();
}

void db::Array::setFilename(const std::string& /*filename*/, 
    const std::string& /*codecname*/) {
  clear(); ///< clear any possible data 
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

#define DIMSWITCH(T,N,D) case N: \
  switch(D) { \
  case 1: return getBzArray<T,1>(m_bzarray); \
  case 2: return getBzArray<T,2>(m_bzarray); \
  case 3: return getBzArray<T,3>(m_bzarray); \
  case 4: return getBzArray<T,4>(m_bzarray); }

void* db::Array::getBlitzArray() const {
  switch(m_elementtype) {
    DIMSWITCH(bool, core::array::t_bool, m_ndim)
    DIMSWITCH(int8_t, core::array::t_int8, m_ndim)
    DIMSWITCH(int16_t, core::array::t_int16, m_ndim)
    DIMSWITCH(int32_t, core::array::t_int32, m_ndim)
    DIMSWITCH(int64_t, core::array::t_int64, m_ndim)
    DIMSWITCH(uint8_t, core::array::t_uint8, m_ndim)
    DIMSWITCH(uint16_t, core::array::t_uint16, m_ndim)
    DIMSWITCH(uint32_t, core::array::t_uint32, m_ndim)
    DIMSWITCH(uint64_t, core::array::t_uint64, m_ndim)
    DIMSWITCH(float, core::array::t_float32, m_ndim)
    DIMSWITCH(float, core::array::t_float32, m_ndim)
    case array::t_float32:
      delete [] static_cast<float*>(m_storage); break;
    case array::t_float64:
      delete [] static_cast<double*>(m_storage); break;
    case array::t_float128:
      delete [] static_cast<long double*>(m_storage); break;
    case array::t_complex64:
      delete [] static_cast<std::complex<float>* >(m_storage); break;
    case array::t_complex128:
      delete [] static_cast<std::complex<double>* >(m_storage); break;
    case array::t_complex256:
      delete [] static_cast<std::complex<long double>* >(m_storage); 
      break;
    default:
      break;
  }
}

void db::Array::deleteBlitzArray() const {
  TDEBUG3("Array destructor (id: " << m_id << ")");
  boost::shared_ptr<const Arrayset> parent(m_parent_arrayset);
  switch(parent->getElementType()) {
    case array::t_bool:
      delete [] static_cast<bool*>(m_storage); break;
    case array::t_int8:
      delete [] static_cast<int8_t*>(m_storage); break;
    case array::t_int16:
      delete [] static_cast<int16_t*>(m_storage); break;
    case array::t_int32:
      delete [] static_cast<int32_t*>(m_storage); break;
    case array::t_int64:
      delete [] static_cast<int64_t*>(m_storage); break;
    case array::t_uint8:
      delete [] static_cast<uint8_t*>(m_storage); break;
    case array::t_uint16:
      delete [] static_cast<uint16_t*>(m_storage); break;
    case array::t_uint32:
      delete [] static_cast<uint32_t*>(m_storage); break;
    case array::t_uint64:
      delete [] static_cast<uint64_t*>(m_storage); break;
    case array::t_float32:
      delete [] static_cast<float*>(m_storage); break;
    case array::t_float64:
      delete [] static_cast<double*>(m_storage); break;
    case array::t_float128:
      delete [] static_cast<long double*>(m_storage); break;
    case array::t_complex64:
      delete [] static_cast<std::complex<float>* >(m_storage); break;
    case array::t_complex128:
      delete [] static_cast<std::complex<double>* >(m_storage); break;
    case array::t_complex256:
      delete [] static_cast<std::complex<long double>* >(m_storage); 
      break;
    default:
      break;
}

blitz::Array<T,D> db::Array::getBlitzArrayWithCast() const {
}

/**
 * Adapts the size of each dimension of the passed blitz array
 * to the ones of the underlying array and copy the data in it.
 */
template<typename T, int D> blitz::Array<T,D> data() const;

/**
 * Adapts the size of each dimension of the passed blitz array
 * to the ones of the underlying array and refer to the data in it.
 * @warning Updating the content of the blitz array will update the
 * content of the corresponding array in the dataset.
 */
template<typename T, int D> blitz::Array<T,D> data();

