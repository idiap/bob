/**
 * @file src/cxx/sp/sp/FFT.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based FFT using Lapack functions
 */

#ifndef TORCH5SPRO_SP_FFT_H
#define TORCH5SPRO_SP_FFT_H

#include "core/logging.h"
#include <blitz/array.h>

namespace Torch {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp {

    /**
     * @brief 1D FFT of a 1D blitz array
     */
    blitz::Array<std::complex<double>,1> 
      fft(const blitz::Array<std::complex<double>,1>& A);

    /**
     * @brief 1D inverse FFT of a 1D blitz array
     */
    blitz::Array<std::complex<double>,1> 
      ifft(const blitz::Array<std::complex<double>,1>& A);

    /**
     * @brief 2D FFT of a 2D blitz array
     */
    blitz::Array<std::complex<double>,2> 
      fft(const blitz::Array<std::complex<double>,2>& A);

    /**
     * @brief 2D inverse FFT of a 2D blitz array
     */
    blitz::Array<std::complex<double>,2> 
      ifft(const blitz::Array<std::complex<double>,2>& A);

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_FFT_H */
