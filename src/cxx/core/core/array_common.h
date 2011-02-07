/**
 * @file src/cxx/core/core/array_common.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file contains information about the supported arrays
 *
 */

#ifndef TORCH_CORE_ARRAY_COMMON_H
#define TORCH_CORE_ARRAY_COMMON_H

#include <stdint.h>
#include <cstdlib>
#include <complex>

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    namespace array {

      /**
       * @brief Enumeration of the supported type for multidimensional arrays
       * @warning float128 and complex256 are defined but currently not 
       * supported
       */
      typedef enum ElementType { 
        t_unknown=0,
        t_bool=1,
        t_int8=2,
        t_int16=3,
        t_int32=4,
        t_int64=5,
        t_uint8=6,
        t_uint16=7,
        t_uint32=8,
        t_uint64=9,
        t_float32=10,
        t_float64=11,
        t_float128=12,
        t_complex64=13,
        t_complex128=14,
        t_complex256=15
      } ElementType;

      /**
       * @brief Maximum number of supported dimensions for multidimensional 
       * arrays.
       */
      const size_t N_MAX_DIMENSIONS_ARRAY = 4;

      /**
       * These are some type to element type conversions
       */
      template<typename T> ElementType getElementType() {
        return t_unknown;
      }

      /**
       * Some specializations that convert type to element type.
       */
      template<> inline ElementType getElementType<bool>() { return t_bool; }
      template<> inline ElementType getElementType<int8_t>() { return t_int8; }
      template<> inline ElementType getElementType<int16_t>() 
      { return t_int16; }
      template<> inline ElementType getElementType<int32_t>() 
      { return t_int32; }
      template<> inline ElementType getElementType<int64_t>() 
      { return t_int64; }
      template<> inline ElementType getElementType<uint8_t>() 
      { return t_uint8; }
      template<> inline ElementType getElementType<uint16_t>() 
      { return t_uint16; }
      template<> inline ElementType getElementType<uint32_t>() 
      { return t_uint32; }
      template<> inline ElementType getElementType<uint64_t>() 
      { return t_uint64; }
      template<> inline ElementType getElementType<float>() 
      { return t_float32; }
      template<> inline ElementType getElementType<double>() 
      { return t_float64; }
      template<> inline ElementType getElementType<long double>() 
      { return t_float128; }
      template<> inline ElementType getElementType<std::complex<float> >() 
      { return t_complex64; }
      template<> inline ElementType getElementType<std::complex<double> >() 
      { return t_complex128; }
      template<> inline ElementType getElementType<std::complex<long double> >()
      { return t_complex256; }

    }

  }
/**
 * @}
 */
}

#endif /* TORCH_CORE_COMMON_ARRAY_H */
