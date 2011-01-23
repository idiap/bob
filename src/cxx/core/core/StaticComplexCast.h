/**
 * @file src/cxx/core/core/StaticComplexCast.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines functions which add std::complex support to the 
 * static_cast function.
 */

#ifndef TORCH5SPRO_CORE_STATIC_COMPLEX_CAST_H
#define TORCH5SPRO_CORE_STATIC_COMPLEX_CAST_H 1

#include <complex>

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    /**
     * @brief Functions which add std::complex support to the static_cast
     * function. This is done by considering the real part only of any
     * complex number.
     */
    template<typename T, typename U> 
    void static_complex_cast(const U& in, T& out) {
      out = static_cast<T>(in);
    }

    template<typename T, typename U> 
    void static_complex_cast(const std::complex<U>& in, T& out) {
      out = static_cast<T>(in.real());
    }

    template<typename T, typename U> 
    void static_complex_cast(const U& in, std::complex<T>& out) {
      out = std::complex<T>(static_cast<T>(in), 0);
    }

    template<typename T, typename U> 
    void static_complex_cast(const std::complex<U>& in, std::complex<T>& out)
    {
      out = std::complex<T>(
        static_cast<T>(in.real()), static_cast<T>(in.imag()) );
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_CORE_STATIC_COMPLEX_CAST_H */

