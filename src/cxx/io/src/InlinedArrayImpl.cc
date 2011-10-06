/**
 * @file src/InlinedArrayImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the InlinedArrayImpl type
 */

#include "io/InlinedArrayImpl.h"
#include "io/Exception.h"

namespace iod = Torch::io::detail;

template<typename T, int D> static inline blitz::Array<T,D>* castBzArray
(void* bzarray) {
  return reinterpret_cast<blitz::Array<T,D>*>(bzarray);
}

template<typename T, int D> static inline void deleteBzArray(void* bzarray) {
  delete castBzArray<T,D>(bzarray);
}

#define DELSWITCH(T,N,D) case N: \
  switch(D) { \
    case 1: deleteBzArray<T,1>(m_bzarray); break;\
    case 2: deleteBzArray<T,2>(m_bzarray); break;\
    case 3: deleteBzArray<T,3>(m_bzarray); break;\
    case 4: deleteBzArray<T,4>(m_bzarray); break;\
    default: throw Torch::io::DimensionError(D,4);\
  } \
  break;

iod::InlinedArrayImpl::~InlinedArrayImpl() {
  switch(m_elementtype) {
    DELSWITCH(bool, core::array::t_bool, m_ndim)
    DELSWITCH(int8_t, core::array::t_int8, m_ndim)
    DELSWITCH(int16_t, core::array::t_int16, m_ndim)
    DELSWITCH(int32_t, core::array::t_int32, m_ndim)
    DELSWITCH(int64_t, core::array::t_int64, m_ndim)
    DELSWITCH(uint8_t, core::array::t_uint8, m_ndim)
    DELSWITCH(uint16_t, core::array::t_uint16, m_ndim)
    DELSWITCH(uint32_t, core::array::t_uint32, m_ndim)
    DELSWITCH(uint64_t, core::array::t_uint64, m_ndim)
    DELSWITCH(float, core::array::t_float32, m_ndim)
    DELSWITCH(double, core::array::t_float64, m_ndim)
    DELSWITCH(long double, core::array::t_float128, m_ndim)
    DELSWITCH(std::complex<float>, core::array::t_complex64, m_ndim)
    DELSWITCH(std::complex<double>, core::array::t_complex128, m_ndim)
    DELSWITCH(std::complex<long double>, core::array::t_complex256, m_ndim)
    default:
      throw Torch::io::TypeError(m_elementtype, Torch::core::array::t_unknown);
  }
}

template<typename T, int D> static inline void* getBzArray(void* bzarray) {
  return reinterpret_cast<void*>(new blitz::Array<T,D>(*castBzArray<T,D>(bzarray)));
}

iod::InlinedArrayImpl::InlinedArrayImpl (const iod::InlinedArrayImpl& other) {
  *this = other; 
}

#define GETSWITCH(T,N,D) case N: \
  switch(D) { \
    case 1: m_bzarray = getBzArray<T,1>(other.m_bzarray); break;\
    case 2: m_bzarray = getBzArray<T,2>(other.m_bzarray); break;\
    case 3: m_bzarray = getBzArray<T,3>(other.m_bzarray); break;\
    case 4: m_bzarray = getBzArray<T,4>(other.m_bzarray); break;\
    default: throw Torch::io::DimensionError(D,4);\
  }\
  break;

iod::InlinedArrayImpl& iod::InlinedArrayImpl::operator= (const iod::InlinedArrayImpl& other) {
  m_elementtype = other.m_elementtype;
  m_ndim = other.m_ndim;
  for (size_t i=0; i<m_ndim; ++i) m_shape[i] = other.m_shape[i];
  switch(m_elementtype) {
    GETSWITCH(bool, core::array::t_bool, m_ndim)
    GETSWITCH(int8_t, core::array::t_int8, m_ndim)
    GETSWITCH(int16_t, core::array::t_int16, m_ndim)
    GETSWITCH(int32_t, core::array::t_int32, m_ndim)
    GETSWITCH(int64_t, core::array::t_int64, m_ndim)
    GETSWITCH(uint8_t, core::array::t_uint8, m_ndim)
    GETSWITCH(uint16_t, core::array::t_uint16, m_ndim)
    GETSWITCH(uint32_t, core::array::t_uint32, m_ndim)
    GETSWITCH(uint64_t, core::array::t_uint64, m_ndim)
    GETSWITCH(float, core::array::t_float32, m_ndim)
    GETSWITCH(double, core::array::t_float64, m_ndim)
    GETSWITCH(long double, core::array::t_float128, m_ndim)
    GETSWITCH(std::complex<float>, core::array::t_complex64, m_ndim)
    GETSWITCH(std::complex<double>, core::array::t_complex128, m_ndim)
    GETSWITCH(std::complex<long double>, core::array::t_complex256, m_ndim)
    default:
      throw Torch::io::TypeError(m_elementtype, Torch::core::array::t_unknown);
  }
  return *this;
}

template<typename T, int D> static void* getBzArrayPtr(void* bzarray) {
  blitz::Array<T,D>* tmp = castBzArray<T,D>(bzarray);
  //normally, the blitz::MemoryBlockReference<T> object does not allow us to
  //steal the data. Here is what we do. We copy the MemoryBlockReference<T>
  //object to a memory area that we create ourselves. Then, we just place a
  //new MemoryBlockReference<T> on that area and this will force the system
  //not to call the destructor for that instance, leaving the memory block
  //undeleted.
  char storage[sizeof(blitz::MemoryBlockReference<T>)];
  new (storage) blitz::MemoryBlockReference<T>(*tmp); //dangle
  return reinterpret_cast<void*>(castBzArray<T,D>(bzarray)->data());
}

#define GETPTRSWITCH(T,N,D) case N: \
  switch(D) { \
    case 1: return getBzArrayPtr<T,1>(m_bzarray); break;\
    case 2: return getBzArrayPtr<T,2>(m_bzarray); break;\
    case 3: return getBzArrayPtr<T,3>(m_bzarray); break;\
    case 4: return getBzArrayPtr<T,4>(m_bzarray); break;\
    default: throw Torch::io::DimensionError(D,4);\
  }\
  break;

void* iod::InlinedArrayImpl::steal_data() {
  switch(m_elementtype) {
    GETPTRSWITCH(bool, core::array::t_bool, m_ndim)
    GETPTRSWITCH(int8_t, core::array::t_int8, m_ndim)
    GETPTRSWITCH(int16_t, core::array::t_int16, m_ndim)
    GETPTRSWITCH(int32_t, core::array::t_int32, m_ndim)
    GETPTRSWITCH(int64_t, core::array::t_int64, m_ndim)
    GETPTRSWITCH(uint8_t, core::array::t_uint8, m_ndim)
    GETPTRSWITCH(uint16_t, core::array::t_uint16, m_ndim)
    GETPTRSWITCH(uint32_t, core::array::t_uint32, m_ndim)
    GETPTRSWITCH(uint64_t, core::array::t_uint64, m_ndim)
    GETPTRSWITCH(float, core::array::t_float32, m_ndim)
    GETPTRSWITCH(double, core::array::t_float64, m_ndim)
    GETPTRSWITCH(long double, core::array::t_float128, m_ndim)
    GETPTRSWITCH(std::complex<float>, core::array::t_complex64, m_ndim)
    GETPTRSWITCH(std::complex<double>, core::array::t_complex128, m_ndim)
    GETPTRSWITCH(std::complex<long double>, core::array::t_complex256, m_ndim)
    default:
      throw Torch::io::TypeError(m_elementtype, Torch::core::array::t_unknown);
  }
  return 0;
}
