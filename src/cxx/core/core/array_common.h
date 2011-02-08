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

      /**
       * These are some type to element size conversions
       */
      template<typename T> size_t getElementSize() {
        return 0;
      }

      /**
       * Some specializations tat convert the types we handle properly
       */
      template<> inline size_t getElementSize<bool>() { return sizeof(bool); }
      template<> inline size_t getElementSize<int8_t>() 
      { return sizeof(int8_t); }
      template<> inline size_t getElementSize<int16_t>() 
      { return sizeof(int16_t); }
      template<> inline size_t getElementSize<int32_t>() 
      { return sizeof(int32_t); }
      template<> inline size_t getElementSize<int64_t>() 
      { return sizeof(int64_t); }
      template<> inline size_t getElementSize<uint8_t>() 
      { return sizeof(uint8_t); }
      template<> inline size_t getElementSize<uint16_t>() 
      { return sizeof(uint16_t); }
      template<> inline size_t getElementSize<uint32_t>() 
      { return sizeof(uint32_t); }
      template<> inline size_t getElementSize<uint64_t>() 
      { return sizeof(uint64_t); }
      template<> inline size_t getElementSize<float>() 
      { return sizeof(float); }
      template<> inline size_t getElementSize<double>() 
      { return sizeof(double); }
      template<> inline size_t getElementSize<long double>() 
      { return sizeof(long double); }
      template<> inline size_t getElementSize<std::complex<float> >() 
      { return sizeof(std::complex<float>); }
      template<> inline size_t getElementSize<std::complex<double> >() 
      { return sizeof(std::complex<double>); }
      template<> inline size_t getElementSize<std::complex<long double> >() 
      { return sizeof(std::complex<long double>); }

      /**
       * Returns the type size given the enumeration
       */
      size_t getElementSize(ElementType t);

    }

  }
/**
 * @}
 */
}

/**
 * Defines the diffType and sizeType in case blitz (old) don't have it defined
 * already.
 */
#ifndef HAVE_BLITZ_SIZETYPE
namespace blitz { typedef int sizeType; }
#endif
#ifndef HAVE_BLITZ_DIFFTYPE
namespace blitz { typedef int diffType; }
#endif

#endif /* TORCH_CORE_COMMON_ARRAY_H */
