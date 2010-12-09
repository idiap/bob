#ifndef TORCH5SPRO_STATIC_COMPLEX_CAST_H
#define TORCH5SPRO_STATIC_COMPLEX_CAST_H 1

#include <complex>

namespace Torch {

  namespace core {

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
}

#endif /* TORCH5SPRO_STATIC_COMPLEX_CAST_H */

