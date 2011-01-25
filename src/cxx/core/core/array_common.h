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
       */
      typedef enum ElementType { t_unknown, t_bool,
        t_int8, t_int16, t_int32, t_int64,
        t_uint8, t_uint16, t_uint32, t_uint64,
        t_float32, t_float64, t_float128,
        t_complex64, t_complex128, t_complex256 } ArrayType;

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

