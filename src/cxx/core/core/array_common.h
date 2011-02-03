/**
 * @file src/cxx/core/core/array_common.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file contains information about the supported arrays
 *
 */

#ifndef TORCH_CORE_ARRAY_COMMON_H
#define TORCH_CORE_ARRAY_COMMON_H

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
      typedef enum ElementType { t_unknown=0, t_bool=1,
        t_int8=2, t_int16=3, t_int32=4, t_int64=5,
        t_uint8=6, t_uint16=7, t_uint32=8, t_uint64=9,
        t_float32=10, t_float64=11, t_float128=12,
        t_complex64=13, t_complex128=14, t_complex256=15 } ArrayType;

      /**
       * @brief Maximum number of supported dimensions for multidimensional 
       * arrays.
       */
      const size_t N_MAX_DIMENSIONS_ARRAY = 4;

    }

  }
/**
 * @}
 */
}

#endif /* TORCH_CORE_COMMON_ARRAY_H */

