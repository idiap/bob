/**
 * @file src/cxx/core/src/Arrayset.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch representation of an Array for a Dataset.
 */

#include "core/Arrayset.h"

#define REFER_DEF(T,name,D) template<>\
  blitz::Array<T,D> Array::refer() \
  { \
    referCheck<D>(); \
    blitz::TinyVector<int,D> shape; \
    m_parent_arrayset.getShape(shape); \
    switch(m_parent_arrayset.getElementType()) { \
      case name: \
        break; \
      default: \
        error << "Cannot refer to data with a " << \
          "blitz array of a different type." << std::endl; \
        throw TypeError(); \
        break; \
    } \
    return blitz::Array<T,D>(reinterpret_cast<T*>(m_storage), \
        shape, blitz::neverDeleteData); \
  } \


namespace Torch {
  namespace core {

    REFER_DEF(bool,array::t_bool,1)
    REFER_DEF(bool,array::t_bool,2)
    REFER_DEF(bool,array::t_bool,3)
    REFER_DEF(bool,array::t_bool,4)
    REFER_DEF(int8_t,array::t_int8,1)
    REFER_DEF(int8_t,array::t_int8,2)
    REFER_DEF(int8_t,array::t_int8,3)
    REFER_DEF(int8_t,array::t_int8,4)
    REFER_DEF(int16_t,array::t_int16,1)
    REFER_DEF(int16_t,array::t_int16,2)
    REFER_DEF(int16_t,array::t_int16,3)
    REFER_DEF(int16_t,array::t_int16,4)
    REFER_DEF(int32_t,array::t_int32,1)
    REFER_DEF(int32_t,array::t_int32,2)
    REFER_DEF(int32_t,array::t_int32,3)
    REFER_DEF(int32_t,array::t_int32,4)
    REFER_DEF(int64_t,array::t_int64,1)
    REFER_DEF(int64_t,array::t_int64,2)
    REFER_DEF(int64_t,array::t_int64,3)
    REFER_DEF(int64_t,array::t_int64,4)
    REFER_DEF(uint8_t,array::t_uint8,1)
    REFER_DEF(uint8_t,array::t_uint8,2)
    REFER_DEF(uint8_t,array::t_uint8,3)
    REFER_DEF(uint8_t,array::t_uint8,4)
    REFER_DEF(uint16_t,array::t_uint16,1)
    REFER_DEF(uint16_t,array::t_uint16,2)
    REFER_DEF(uint16_t,array::t_uint16,3)
    REFER_DEF(uint16_t,array::t_uint16,4)
    REFER_DEF(uint32_t,array::t_uint32,1)
    REFER_DEF(uint32_t,array::t_uint32,2)
    REFER_DEF(uint32_t,array::t_uint32,3)
    REFER_DEF(uint32_t,array::t_uint32,4)
    REFER_DEF(uint64_t,array::t_uint64,1)
    REFER_DEF(uint64_t,array::t_uint64,2)
    REFER_DEF(uint64_t,array::t_uint64,3)
    REFER_DEF(uint64_t,array::t_uint64,4)
    REFER_DEF(float,array::t_float32,1)
    REFER_DEF(float,array::t_float32,2)
    REFER_DEF(float,array::t_float32,3)
    REFER_DEF(float,array::t_float32,4)
    REFER_DEF(double,array::t_float64,1)
    REFER_DEF(double,array::t_float64,2)
    REFER_DEF(double,array::t_float64,3)
    REFER_DEF(double,array::t_float64,4)
    REFER_DEF(std::complex<float>,array::t_complex64,1)
    REFER_DEF(std::complex<float>,array::t_complex64,2)
    REFER_DEF(std::complex<float>,array::t_complex64,3)
    REFER_DEF(std::complex<float>,array::t_complex64,4)
    REFER_DEF(std::complex<double>,array::t_complex128,1)
    REFER_DEF(std::complex<double>,array::t_complex128,2)
    REFER_DEF(std::complex<double>,array::t_complex128,3)
    REFER_DEF(std::complex<double>,array::t_complex128,4)

  }
}

