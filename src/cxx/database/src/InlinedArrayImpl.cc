/**
 * @file src/InlinedArrayImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the InlinedArrayImpl type
 */

#include "database/InlinedArrayImpl.h"
#include "database/dataset_common.h"
#include "database/Exception.h"

namespace tdd = Torch::database::detail;

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
    default: throw Torch::database::DimensionError(D,4);\
  } \
  break;

tdd::InlinedArrayImpl::~InlinedArrayImpl() {
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
      throw Torch::database::TypeError(m_elementtype, Torch::core::array::t_unknown);
  }
}

template<typename T, int D> static inline void* getBzArray(void* bzarray) {
  return reinterpret_cast<void*>(new blitz::Array<T,D>(*castBzArray<T,D>(bzarray)));
}

tdd::InlinedArrayImpl::InlinedArrayImpl (const tdd::InlinedArrayImpl& other) {
  *this = other; 
}

#define GETSWITCH(T,N,D) case N: \
  switch(D) { \
    case 1: m_bzarray = getBzArray<T,1>(other.m_bzarray); break;\
    case 2: m_bzarray = getBzArray<T,2>(other.m_bzarray); break;\
    case 3: m_bzarray = getBzArray<T,3>(other.m_bzarray); break;\
    case 4: m_bzarray = getBzArray<T,4>(other.m_bzarray); break;\
    default: throw Torch::database::DimensionError(D,4);\
  }\
  break;

tdd::InlinedArrayImpl& tdd::InlinedArrayImpl::operator= (const tdd::InlinedArrayImpl& other) {
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
      throw Torch::database::TypeError(m_elementtype, Torch::core::array::t_unknown);
  }
  return *this;
}
