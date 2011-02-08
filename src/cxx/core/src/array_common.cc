/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue  8 Feb 16:35:57 2011 
 *
 * @brief Some common array utilities 
 */

#include "core/array_common.h"
#include "core/Exception.h"

size_t Torch::core::array::getElementSize(ElementType t) {
  switch(t) {
    case Torch::core::array::t_bool:
      return sizeof(bool);
    case Torch::core::array::t_int8:
      return sizeof(int8_t);
    case Torch::core::array::t_int16:
      return sizeof(int16_t);
    case Torch::core::array::t_int32:
      return sizeof(int32_t);
    case Torch::core::array::t_int64:
      return sizeof(int64_t);
    case Torch::core::array::t_uint8:
      return sizeof(uint8_t);
    case Torch::core::array::t_uint16:
      return sizeof(uint16_t);
    case Torch::core::array::t_uint32:
      return sizeof(uint32_t);
    case Torch::core::array::t_uint64:
      return sizeof(uint64_t);
    case Torch::core::array::t_float32:
      return sizeof(float);
    case Torch::core::array::t_float64:
      return sizeof(double);
    case Torch::core::array::t_float128:
      return sizeof(long double);
    case Torch::core::array::t_complex64:
      return sizeof(std::complex<float>);
    case Torch::core::array::t_complex128:
      return sizeof(std::complex<double>);
    case Torch::core::array::t_complex256:
      return sizeof(std::complex<long double>);
    default:
      throw Exception();
  }
}
