/**
 * @file src/cxx/sp/sp/FFT.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based FFT using Lapack functions
 */

#ifndef TORCH5SPRO_SP_FFT_H
#define TORCH5SPRO_SP_FFT_H

#include "core/array_common.h"
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
    void fft(const blitz::Array<std::complex<double>,1>& A, 
      blitz::Array<std::complex<double>,1>& B);

    /**
     * @brief 1D inverse FFT of a 1D blitz array
     */
    void ifft(const blitz::Array<std::complex<double>,1>& A,
      blitz::Array<std::complex<double>,1>& B);

    /**
     * @brief 2D FFT of a 2D blitz array
     */
    void fft(const blitz::Array<std::complex<double>,2>& A,
      blitz::Array<std::complex<double>,2>& B);

    /**
     * @brief 2D inverse FFT of a 2D blitz array
     */
    void ifft(const blitz::Array<std::complex<double>,2>& A,
      blitz::Array<std::complex<double>,2>& B);


    /**
     * @brief Rearrange the output of the fft by moving the zero-frequency 
     * component to the center of the 1D blitz array. 
     */
    void fftshift(const blitz::Array<std::complex<double>,1>& A,
      blitz::Array<std::complex<double>,1>& B);

    /**
     * @brief Swap the two halves of the 1D blitz array wrt. to its center
     * ifftshift reverts the result of fftshift, which is important for 
     * dimensions of odd size.
     */
    void ifftshift(const blitz::Array<std::complex<double>,1>& A,
      blitz::Array<std::complex<double>,1>& B);

    /**
     * @brief Rearrange the output of the fft by moving the zero-frequency 
     * component to the center of the 2D blitz array. Therefore, it swaps 
     * the first quadrant with the third and the second quadrant with the 
     * fourth.
     */
    void fftshift(const blitz::Array<std::complex<double>,2>& A,
      blitz::Array<std::complex<double>,2>& B);

    /**
     * @brief Swap the first quadrant with the third and the second quadrant 
     * with the fourth. ifftshift reverts the result of fftshift, which is 
     * important for dimensions of odd size.
     */
    void ifftshift(const blitz::Array<std::complex<double>,2>& A,
      blitz::Array<std::complex<double>,2>& B);

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_FFT_H */
