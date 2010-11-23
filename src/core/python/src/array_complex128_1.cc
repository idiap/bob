/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @brief blitz::Array<std::complex<double>,1> to and from python converters
 */
#include "core/python/array.h"
declare_complex_array(std::complex<double>, 1, complex128, bind_core_array_complex128_1)
