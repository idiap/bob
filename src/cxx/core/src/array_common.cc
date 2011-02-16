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

const char* Torch::core::array::stringize(ElementType t) {
  static const char* t_bool_string = "bool";
  static const char* t_int8_string = "int8";
  static const char* t_int16_string = "int16";
  static const char* t_int32_string = "int32";
  static const char* t_int64_string = "int64";
  static const char* t_uint8_string = "uint8";
  static const char* t_uint16_string = "uint16";
  static const char* t_uint32_string = "uint32";
  static const char* t_uint64_string = "uint64";
  static const char* t_float32_string = "float32";
  static const char* t_float64_string = "float64";
  static const char* t_float128_string = "float128";
  static const char* t_complex64_string = "complex64";
  static const char* t_complex128_string = "complex128";
  static const char* t_complex256_string = "complex256";
  static const char* t_unknown_string = "unknown";
  switch(t) {
    case Torch::core::array::t_bool: 
      return t_bool_string;
    case Torch::core::array::t_int8:
      return t_int8_string;
    case Torch::core::array::t_int16:
      return t_int16_string;
    case Torch::core::array::t_int32:
      return t_int32_string;
    case Torch::core::array::t_int64:
      return t_int64_string;
    case Torch::core::array::t_uint8:
      return t_uint8_string;
    case Torch::core::array::t_uint16:
      return t_uint16_string;
    case Torch::core::array::t_uint32:
      return t_uint32_string;
    case Torch::core::array::t_uint64:
      return t_uint64_string;
    case Torch::core::array::t_float32:
      return t_float32_string;
    case Torch::core::array::t_float64:
      return t_float64_string;
    case Torch::core::array::t_float128:
      return t_float128_string;
    case Torch::core::array::t_complex64:
      return t_complex64_string;
    case Torch::core::array::t_complex128:
      return t_complex128_string;
    case Torch::core::array::t_complex256:
      return t_complex256_string;
    default:
      return t_unknown_string;
  }
}
