/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue  8 Feb 16:35:57 2011 
 *
 * @brief Some common array utilities 
 */

#include "core/array_common.h"

size_t Torch::core::array::getElementSize(ElementType t) {
  switch(t) {
    case Torch::core::array::t_bool:
      data_size = sizeof(bool); break;
    case Torch::core::array::t_int8:
      data_size = sizeof(int8_t); break;
    case Torch::core::array::t_int16:
      data_size = sizeof(int16_t); break;
    case Torch::core::array::t_int32:
      data_size = sizeof(int32_t); break;
    case Torch::core::array::t_int64:
      data_size = sizeof(int64_t); break;
    case Torch::core::array::t_uint8:
      data_size = sizeof(uint8_t); break;
    case Torch::core::array::t_uint16:
      data_size = sizeof(uint16_t); break;
    case Torch::core::array::t_uint32:
      data_size = sizeof(uint32_t); break;
    case Torch::core::array::t_uint64:
      data_size = sizeof(uint64_t); break;
    case Torch::core::array::t_float32:
      data_size = sizeof(float); break;
    case Torch::core::array::t_float64:
      data_size = sizeof(double); break;
    case Torch::core::array::t_float128:
      data_size = sizeof(long double); break;
    case Torch::core::array::t_complex64:
      data_size = sizeof(std::complex<float>); break;
    case Torch::core::array::t_complex128:
      data_size = sizeof(std::complex<double>); break;
    case Torch::core::array::t_complex256:
      data_size = sizeof(std::complex<long double>); break;
      throw Exception();
      break;
  }
}
