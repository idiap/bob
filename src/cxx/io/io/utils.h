/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  6 Oct 11:09:39 2011 
 *
 * @brief Utilities for converting data to-from blitz::Arrays and other
 * goodies.
 */

#ifndef TORCH_IO_UTILS_H 
#define TORCH_IO_UTILS_H

#include <blitz/array.h>
#include <stdint.h>
#include <stdexcept>

#include "io/buffer.h"
#include "core/cast.h"

namespace Torch { namespace io {

  /**
   * Fills in shape and stride starting from a typeinfo object
   */
  template <int N> void set_shape_and_stride(const typeinfo& info,
      blitz::TinyVector<int,N>& shape, blitz::TinyVector<int,N>& stride) {
    for (int k=0; k<N; ++k) {
      shape[k] = info.shape[k];
      stride[k] = info.stride[k];
    }
  }


  /**
   * Takes a data pointer and assumes it is a C-style array for the defined
   * type. Creates a wrapper as a blitz::Array<T,N> with the same number of
   * dimensions and type. Notice that the blitz::Array<> created will have its
   * memory tied to the passed buffer. In other words you have to make sure
   * that the buffer outlives the returned blitz::Array<>.
   */
  template <typename T, int N>
  blitz::Array<T,N> wrap(const io::buffer& buf) {
    
    const typeinfo& type = buf.type();

    if (!buf.ptr()) throw std::runtime_error("empty buffer");

    if (type.dtype != Torch::core::array::getElementType<T>()) throw std::runtime_error("cannot efficiently retrieve blitz::Array<> from buffer containing a different dtype");

    if (type.nd != N) throw std::runtime_error("cannot retrieve blitz::Array<> from buffer containing different dimensionality");

    blitz::TinyVector<int,N> shape;
    blitz::TinyVector<int,N> stride;
    set_shape_and_stride(type, shape, stride);

    return blitz::Array<T,N>((T*)buf.ptr(), 
        shape, stride, blitz::neverDeleteData);
  }


  /**
   * Takes a data pointer and assumes it is a C-style array for the defined
   * type. Creates a copy as a blitz::Array<T,N> with the same number of
   * dimensions, but with a type as specified by you. If the type does not
   * match the type of the original C-style array, a cast will happen. 
   *
   * If a certain type cast is not supported. An appropriate exception will
   * be raised.
   */
  template <typename T, int N>
  blitz::Array<T,N> cast(const io::buffer& buf) {

    const typeinfo& type = buf.type();

    if (type.nd != N) throw std::runtime_error("cannot cast blitz::Array<> from buffer containing different dimensionality");

    switch (type.dtype) {
      case Torch::core::array::t_bool: 
        return Torch::core::cast<T>(wrap<bool,N>(buf));
      case Torch::core::array::t_int8: 
        return Torch::core::cast<T>(wrap<int8_t,N>(buf));
      case Torch::core::array::t_int16: 
        return Torch::core::cast<T>(wrap<int16_t,N>(buf));
      case Torch::core::array::t_int32: 
        return Torch::core::cast<T>(wrap<int32_t,N>(buf));
      case Torch::core::array::t_int64: 
        return Torch::core::cast<T>(wrap<int64_t,N>(buf));
      case Torch::core::array::t_uint8: 
        return Torch::core::cast<T>(wrap<uint8_t,N>(buf));
      case Torch::core::array::t_uint16: 
        return Torch::core::cast<T>(wrap<uint16_t,N>(buf));
      case Torch::core::array::t_uint32: 
        return Torch::core::cast<T>(wrap<uint32_t,N>(buf));
      case Torch::core::array::t_uint64: 
        return Torch::core::cast<T>(wrap<uint64_t,N>(buf));
      case Torch::core::array::t_float32: 
        return Torch::core::cast<T>(wrap<float,N>(buf));
      case Torch::core::array::t_float64: 
        return Torch::core::cast<T>(wrap<double,N>(buf));
      case Torch::core::array::t_float128: 
        return Torch::core::cast<T>(wrap<long double,N>(buf));
      case Torch::core::array::t_complex64: 
        return Torch::core::cast<T>(wrap<std::complex<float>,N>(buf));
      case Torch::core::array::t_complex128: 
        return Torch::core::cast<T>(wrap<std::complex<double>,N>(buf));
      case Torch::core::array::t_complex256: 
        return Torch::core::cast<T>(wrap<std::complex<long double>,N>(buf));
      default:
        break;
  }

    //if we get to this point, there is nothing much we can do...
    throw std::runtime_error("invalid type on blitz buffer array casting -- debug me");

  }

}}

#endif /* TORCH_IO_UTILS_H */

