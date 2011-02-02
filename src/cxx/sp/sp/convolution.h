/**
 * @file src/cxx/sp/sp/convolution.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based convolution product with zero padding
 */

#ifndef TORCH5SPRO_SP_CONVOLVE_H
#define TORCH5SPRO_SP_CONVOLVE_H

#include "core/logging.h"
#include "core/Exception.h"
#include <blitz/array.h>

namespace Torch {
  /**
   * \ingroup libsp_api
   * @{
   *
   */
  namespace sp {

    /**
     * @brief Enumeration of the possible options
     */
    enum ConvolutionOption {
      FULL,
      SAME,
      VALID
    };

    /**
     * @brief 1D convolution of blitz arrays using zero padding
     * @param option:  * 0: full size (default)
     *                 * 1: same size as B
     *                 * 2: valid (part without zero padding)
     * @warning Assume size(A) > size(B)
     */
    template<typename T>
      blitz::Array<T,1> convolve(const blitz::Array<T,1>& B, 
        const blitz::Array<T,1>& C, const ConvolutionOption option = FULL);

    /**
     * @brief 2D convolution of blitz arrays using zero padding
     * @param option:  * 0: full size
     *                 * 1: same size as B
     *                 * 2: valid (part without zero padding)
     * @warning Assume size(A) > size(B)
     */
    template<typename T>
      blitz::Array<T,2> convolve(const blitz::Array<T,2>& B, 
        const blitz::Array<T,2>& C, const ConvolutionOption option = FULL);

  }
/**
 * @}
 */
}

#include "sp/convolution.cc"

#endif /* TORCH5SPRO_SP_CONVOLVE_H */

