/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 16 Sep 2011 14:03:41 CEST
 *
 * @brief A macro to find out the current endianness of the system
 */

#ifndef TORCH_CORE_PYTHON_ENDIANNESS_H 
#define TORCH_CORE_PYTHON_ENDIANNESS_H

#include <limits.h>
#include <stdint.h>

#if CHAR_BIT != 8
#error "unsupported char size"
#endif

namespace Torch { namespace python {

  enum Endianness {
    TORCH_LITTLE_ENDIAN = 0x03020100ul,
    TORCH_BIG_ENDIAN = 0x00010203ul,
    TORCH_PDP_ENDIAN = 0x01000302ul
  };

  static const union { 
    unsigned char bytes[4]; 
    uint32_t value; 
  } torch_host_order = { { 0, 1, 2, 3 } };

  inline enum Endianness getEndianness() { 
    return static_cast<enum Endianness>(torch_host_order.value); 
  }

}}

#endif /* TORCH_CORE_PYTHON_ENDIANNESS_H */

